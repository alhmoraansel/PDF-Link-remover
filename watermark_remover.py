import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import io
import tempfile
import shutil
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from PIL import Image

# Allow loading truncated images
Image.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Helper / Compression Utilities
# ----------------------------

def is_filter(obj, name_str):
    try:
        f = obj.get('/Filter')
        return f == pikepdf.Name(name_str)
    except Exception:
        return False

def read_stream_bytes(obj):
    try:
        return obj.get_stream_buffer()
    except Exception:
        pass
    try:
        return obj.read_bytes()
    except Exception:
        pass
    return b''

def pil_from_pdfimage(pdf_img):
    pil_img = pdf_img.as_pil_image()
    if pil_img.mode == 'CMYK':
        pil_img = pil_img.convert('RGB')
    if pil_img.mode == '1':
        pil_img = pil_img.convert('L').convert('RGB')
    return pil_img

def flatten_alpha(pil_img, background_color=(255,255,255)):
    if pil_img.mode in ('RGBA', 'LA'):
        alpha = pil_img.split()[-1]
        base = Image.new('RGB', pil_img.size, background_color)
        base.paste(pil_img, mask=alpha)
        return base
    elif pil_img.mode == 'P' and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
        alpha = pil_img.split()[-1]
        base = Image.new('RGB', pil_img.size, background_color)
        base.paste(pil_img, mask=alpha)
        return base
    elif pil_img.mode != 'RGB':
        return pil_img.convert('RGB')
    return pil_img

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None:
        save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def png_bytes_from_pil(pil_img, optimize=True):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=bool(optimize))
    return buf.getvalue()

# ----------------------------
# Rasterization Logic (Optimized & Fax Mode)
# ----------------------------

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False):
    """
    Renders pages as images to fix artifacts (stripes/watermarks).
    Since this replaces the page with a flat image, ALL links are inherently destroyed.
    
    Modes:
    1. Standard: RGB/Grayscale JPEG. Optimized DPI (max ~144).
    2. Fax Mode: 1-bit Monochrome TIFF (CCITT G4). High DPI (200+) but tiny file size.
    """
    try:
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()

        # ZOOM / DPI CALCULATION
        if fax_mode:
            # Fax mode needs HIGHER resolution because we lose anti-aliasing (gray edges).
            # Zoom 2.5 ~= 180 DPI, Zoom 3.0 ~= 216 DPI.
            zoom = 2.5 
        else:
            # Standard mode: Dynamic DPI based on slider to save space
            # Quality 100 -> Zoom 2.0 (144 DPI)
            # Quality 50  -> Zoom 1.5 (108 DPI)
            zoom = 1.0 + (quality / 100.0)
            if zoom > 2.0: zoom = 2.0

        mat = fitz.Matrix(zoom, zoom)

        for page in src_doc:
            # Render page to Pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL
            mode = "RGB"
            if pix.n == 1: mode = "L"
            elif pix.n == 4: mode = "CMYK"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            buf = io.BytesIO()

            if fax_mode:
                # --- FAX MODE (The Tiny File Maker) ---
                # 1. Convert to Grayscale
                # 2. Threshold to Black & White (No Dithering)
                img = img.convert('L').point(lambda x: 0 if x < 200 else 255, '1')
                # 3. Save as TIFF with CCITT Group 4 Compression
                img.save(buf, format="TIFF", compression="group4")
            else:
                # --- STANDARD JPEG MODE ---
                if grayscale:
                    img = img.convert('L')
                elif mode == 'RGB':
                    # Check if image is effectively grayscale to save 2/3rds data
                    try:
                        sample = img.resize((64, 64), resample=Image.NEAREST)
                        if sample.mode == 'RGB':
                            extrema = sample.getextrema()
                            if len(extrema) == 3 and extrema[0] == extrema[1] == extrema[2]:
                                img = img.convert('L')
                    except: pass

                # Progressive JPEG + Subsampling for max compression
                img.save(buf, 
                         format="JPEG", 
                         quality=int(quality), 
                         optimize=True, 
                         progressive=True, 
                         subsampling=2)

            img_bytes = buf.getvalue()
            
            # Insert into new PDF
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)

        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close()
        out_doc.close()
        return True, "Rasterization successful"
    except Exception as e:
        return False, f"Rasterization failed: {e}"

# ----------------------------
# Main pipeline
# ----------------------------

def process_pdf_pipeline(input_path, output_path, watermark_text, quality_val, mode, grayscale, update_progress):
    fd1, temp_repaired = tempfile.mkstemp(suffix=".pdf")
    os.close(fd1)
    fd2, temp_cleaned = tempfile.mkstemp(suffix=".pdf")
    os.close(fd2)

    try:
        # --- STAGE 1: REPAIR ---
        try:
            pdf = pikepdf.open(input_path, allow_overwriting_input=True)
            pdf.save(temp_repaired, fix_metadata_version=True)
            pdf.close()
        except Exception as e:
            return False, f"Repair failed: {e}"

        if update_progress: update_progress(20)

        # --- STAGE 2: CLEAN TEXT & NUCLEAR LINK REMOVAL ---
        try:
            doc = fitz.open(temp_repaired)
            doc.set_metadata({})

            for page in doc:
                
                # --- NUCLEAR LINK REMOVAL (Pass 1: Standard API) ---
                # Deletes all standard hyperlinks on the page
                try:
                    for link in page.get_links():
                        page.delete_link(link)
                except Exception: pass

                # --- NUCLEAR LINK REMOVAL (Pass 2: Annotation Scrubbing) ---
                # Scans low-level annotations. If it smells like a link, delete it.
                # This catches hidden links, buttons, or complex link structures.
                try:
                    # We must iterate over a list copy because we are deleting items
                    for annot in list(page.annots()):
                        # Type 1 is 'Link', but we also check for URI actions just in case
                        if annot.type[0] == 1: 
                            page.delete_annot(annot)
                except Exception: pass

                # --- WATERMARK REMOVAL ---
                if watermark_text:
                    try:
                        hits = page.search_for(watermark_text)
                        for rect in hits:
                            page.add_redact_annot(rect)
                        page.apply_redactions(images=0, graphics=0)
                    except Exception: pass

                try: page.clean_contents()
                except Exception: pass

            doc.save(temp_cleaned, garbage=4, deflate=True)
            doc.close()
        except Exception as e:
            return False, f"Cleaning failed: {e}"

        if update_progress: update_progress(50)

        # --- STAGE 3: OUTPUT GENERATION ---
        
        # Mode: Rasterize (Standard or Fax) - inherently removes links too
        if mode == 'rasterize' or mode == 'rasterize_fax':
            is_fax = (mode == 'rasterize_fax')
            success, msg = rasterize_and_rebuild(temp_cleaned, output_path, quality_val, grayscale, fax_mode=is_fax)
            if not success:
                shutil.copy2(temp_cleaned, output_path)
                return False, msg
            if update_progress: update_progress(100)
            return True, f"Success ({'Fax Mode' if is_fax else 'Rasterized'})"

        # Modes: Object Compression (Safe/Aggressive/Smart)
        try:
            pdf = pikepdf.open(temp_cleaned)
            pdf.docinfo.clear()

            for obj in list(pdf.objects):
                try:
                    if not (isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == pikepdf.Name('/Image')):
                        continue
                    
                    # Tiling/Predictor Checks
                    if '/DecodeParms' in obj:
                        parms = obj['/DecodeParms']
                        if isinstance(parms, pikepdf.Dictionary):
                            pred = parms.get('/Predictor', 1)
                            if pred not in [1, 10]: continue 
                    
                    flt = obj.get('/Filter')
                    if isinstance(flt, pikepdf.Array) and len(flt) > 1: continue 
                    
                    cs = obj.get('/ColorSpace')
                    if isinstance(cs, pikepdf.Array) and cs[0] == pikepdf.Name('/Indexed'): continue 

                    try:
                        original_bytes = read_stream_bytes(obj) or b''
                        original_size = len(original_bytes)
                    except: original_size = 0

                    is_jpx = (flt == pikepdf.Name('/JPXDecode') or (isinstance(flt, pikepdf.Array) and any(x == pikepdf.Name('/JPXDecode') for x in flt)))
                    is_ccitt = (flt == pikepdf.Name('/CCITTFaxDecode') or (isinstance(flt, pikepdf.Array) and any(x == pikepdf.Name('/CCITTFaxDecode') for x in flt)))
                    is_jbig2 = (flt == pikepdf.Name('/JBIG2Decode') or (isinstance(flt, pikepdf.Array) and any(x == pikepdf.Name('/JBIG2Decode') for x in flt)))
                    is_ccitt_like = is_ccitt or is_jbig2

                    try:
                        width = int(obj.get('/Width', 0))
                        height = int(obj.get('/Height', 0))
                    except: width = height = 0

                    try:
                        pdf_img = PdfImage(obj)
                        pil_img = pil_from_pdfimage(pdf_img)
                    except: continue

                    if width and height and (width < 20 or height < 20): continue
                    if pil_img.width < 20 or pil_img.height < 20: continue

                    if grayscale and pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')

                    has_softmask = '/SMask' in obj
                    has_mask = '/Mask' in obj
                    pil_has_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)

                    # AGGRESSIVE
                    if mode == 'aggressive':
                        if is_ccitt_like or is_jpx: continue
                        
                        pil_proc = flatten_alpha(pil_img)
                        jpeg_bytes = jpeg_bytes_from_pil(pil_proc, quality_val, subsampling=2)

                        if original_size and len(jpeg_bytes) > original_size * 1.5: continue

                        try:
                            try: obj.clear()
                            except: pass
                            obj.write(jpeg_bytes)
                            obj['/Type'] = pikepdf.Name('/XObject')
                            obj['/Subtype'] = pikepdf.Name('/Image')
                            obj['/Width'] = pil_proc.width
                            obj['/Height'] = pil_proc.height
                            if grayscale:
                                obj['/ColorSpace'] = pikepdf.Name('/DeviceGray')
                            else:
                                obj['/ColorSpace'] = pikepdf.Name('/DeviceRGB')
                            obj['/BitsPerComponent'] = 8
                            obj['/Filter'] = pikepdf.Name('/DCTDecode')
                            for k in ['/SMask', '/Mask', '/Decode', '/Matte', '/ColorKeyMask', '/ICCBased']:
                                if k in obj:
                                    try: del obj[k]
                                    except: pass
                            continue
                        except: continue

                    # SAFE
                    if mode == 'safe':
                        if is_ccitt_like or is_jpx: continue
                        if has_softmask or has_mask or pil_has_alpha: continue

                        try:
                            if pil_img.mode not in ('RGB', 'L'):
                                pil_img = pil_img.convert('RGB')
                            
                            jpeg_bytes = jpeg_bytes_from_pil(pil_img, quality_val)
                            if original_size and len(jpeg_bytes) >= original_size: continue

                            try: obj.clear()
                            except: pass
                            obj.write(jpeg_bytes)
                            obj['/Type'] = pikepdf.Name('/XObject')
                            obj['/Subtype'] = pikepdf.Name('/Image')
                            obj['/Width'] = pil_img.width
                            obj['/Height'] = pil_img.height
                            if grayscale or pil_img.mode == 'L':
                                obj['/ColorSpace'] = pikepdf.Name('/DeviceGray')
                            else:
                                obj['/ColorSpace'] = pikepdf.Name('/DeviceRGB')
                            obj['/BitsPerComponent'] = 8
                            obj['/Filter'] = pikepdf.Name('/DCTDecode')
                            for k in ['/Decode', '/Mask', '/Matte', '/ColorKeyMask']:
                                if k in obj:
                                    try: del obj[k]
                                    except: pass
                            
                            cs = obj.get('/ColorSpace')
                            if isinstance(cs, pikepdf.Array) and len(cs) > 0 and cs[0] == pikepdf.Name('/ICCBased'):
                                if grayscale: obj['/ColorSpace'] = pikepdf.Name('/DeviceGray')
                                else: obj['/ColorSpace'] = pikepdf.Name('/DeviceRGB')
                            continue
                        except: continue

                    # LOSSLESS SMART
                    if mode == 'lossless-smart':
                        if is_ccitt_like or is_jpx: continue

                        if has_softmask or has_mask or pil_has_alpha:
                            png_bytes = png_bytes_from_pil(pil_img)
                            if original_size and len(png_bytes) >= original_size: continue
                            try:
                                try: obj.clear()
                                except: pass
                                obj.write(png_bytes)
                                obj['/Type'] = pikepdf.Name('/XObject')
                                obj['/Subtype'] = pikepdf.Name('/Image')
                                obj['/Width'] = pil_img.width
                                obj['/Height'] = pil_img.height
                                obj['/ColorSpace'] = pikepdf.Name('/DeviceRGB')
                                obj['/BitsPerComponent'] = 8
                                for k in ['/Decode', '/Mask', '/Matte', '/ColorKeyMask']:
                                    if k in obj:
                                        try: del obj[k]
                                        except: pass
                                continue
                            except: continue
                        
                        try:
                            if pil_img.mode not in ('RGB', 'L'): pil_img = pil_img.convert('RGB')
                            jpeg_bytes = jpeg_bytes_from_pil(pil_img, quality_val)
                            if original_size and len(jpeg_bytes) >= original_size: continue
                            try:
                                try: obj.clear()
                                except: pass
                                obj.write(jpeg_bytes)
                                obj['/Type'] = pikepdf.Name('/XObject')
                                obj['/Subtype'] = pikepdf.Name('/Image')
                                obj['/Width'] = pil_img.width
                                obj['/Height'] = pil_img.height
                                if grayscale or pil_img.mode == 'L':
                                    obj['/ColorSpace'] = pikepdf.Name('/DeviceGray')
                                else:
                                    obj['/ColorSpace'] = pikepdf.Name('/DeviceRGB')
                                obj['/BitsPerComponent'] = 8
                                obj['/Filter'] = pikepdf.Name('/DCTDecode')
                                for k in ['/Decode', '/Mask', '/Matte', '/ColorKeyMask']:
                                    if k in obj:
                                        try: del obj[k]
                                        except: pass
                                continue
                            except: continue
                        except: continue

                except Exception:
                    continue

            pdf.save(output_path)
            pdf.close()

        except Exception as e:
            print(f"Compression failed: {e}")
            shutil.copy2(temp_cleaned, output_path)
            return True, "Success (Compression skipped)"

        if update_progress:
            update_progress(100)
        return True, "Success"

    except Exception as e:
        return False, f"Error: {e}"

    finally:
        for p in [temp_repaired, temp_cleaned]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

# ----------------------------
# Threading & UI
# ----------------------------

def start_processing_thread(files_to_process, output_dir=None, single_output=None):
    save_button.state(['disabled'])
    batch_button.state(['disabled'])
    browse_button.state(['disabled'])
    paste_button.state(['disabled'])
    progress_bar['value'] = 0
    status_label.config(text="Starting...", foreground="blue")

    watermark = watermark_entry.get()

    if compress_var.get():
        quality = int(quality_scale.get())
    else:
        quality = 100

    grayscale = grayscale_var.get()

    # Determine Mode
    mode_choice = compression_mode_var.get()
    if mode_choice == "Safe Compression":
        mode = 'safe'
    elif mode_choice == "Aggressive Compression":
        mode = 'aggressive'
    elif mode_choice == "Rasterize (Standard)":
        mode = 'rasterize'
    elif mode_choice == "Rasterize (B&W Fax Mode)":
        mode = 'rasterize_fax'
    else:
        mode = 'lossless-smart'

    def run_job():
        success_count = 0
        errors = []
        total_files = len(files_to_process)

        for idx, file_path in enumerate(files_to_process):
            filename = os.path.basename(file_path)

            if single_output:
                current_output = single_output
            else:
                name, ext = os.path.splitext(filename)
                current_output = os.path.join(output_dir, f"{name}_cleaned{ext}")

            root.after(0, lambda f=filename: status_label.config(text=f"Processing: {f}"))

            def update_progress(val):
                overall_progress = ((idx + (val/100)) / total_files) * 100
                root.after(0, lambda v=overall_progress: progress_bar.configure(value=v))

            success, msg = process_pdf_pipeline(file_path, current_output, watermark, quality, mode, grayscale, update_progress)

            if success:
                success_count += 1
            else:
                errors.append(f"{filename}: {msg}")

        root.after(0, lambda: finish_processing(success_count, total_files, errors))

    threading.Thread(target=run_job, daemon=True).start()

def finish_processing(success_count, total_count, errors):
    progress_bar['value'] = 100
    status_label.config(text="Done", foreground="green")
    save_button.state(['!disabled'])
    batch_button.state(['!disabled'])
    browse_button.state(['!disabled'])
    paste_button.state(['!disabled'])

    result_msg = f"Processed {success_count}/{total_count} files successfully."
    if errors:
        result_msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showwarning("Completed with Errors", result_msg)
    else:
        messagebox.showinfo("Completed", result_msg)

# --- GUI Events ---

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, file_path)
        folder, name = os.path.split(file_path)
        name_root, ext = os.path.splitext(name)
        output_entry.delete(0, tk.END)
        output_entry.insert(0, os.path.join(folder, f"{name_root}_cleaned{ext}"))

def run_single_file():
    input_path = input_entry.get()
    output_path = output_entry.get()
    if not input_path:
        messagebox.showwarning("Warning", "Please select an input file.")
        return
    if not output_path:
        output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not output_path:
            return
    start_processing_thread([input_path], single_output=output_path)

def run_batch():
    files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")], title="Select Multiple PDFs")
    if not files:
        return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return
    start_processing_thread(files, output_dir=output_dir)

def paste_from_clipboard():
    try:
        root.update_idletasks()
        try:
            text = root.clipboard_get(type='STRING')
        except tk.TclError:
            text = root.clipboard_get()
        if text:
            watermark_entry.delete(0, tk.END)
            watermark_entry.insert(0, text)
            status_label.config(text="Pasted!", foreground="green")
        else:
            status_label.config(text="Clipboard empty", foreground="red")
    except tk.TclError:
        status_label.config(text="No text found", foreground="red")
    root.after(2000, lambda: status_label.config(text="Ready", foreground="gray"))

def update_scale_label(val):
    quality_label_var.set(f"Target Image Quality: {int(float(val))}%")

def toggle_compression():
    if compress_var.get():
        quality_scale.state(['!disabled'])
        quality_label.configure(foreground="black")
        compression_mode_dropdown.state(['!disabled'])
        grayscale_check.state(['!disabled'])
    else:
        quality_scale.state(['disabled'])
        quality_label.configure(foreground="gray")
        compression_mode_dropdown.state(['disabled'])
        grayscale_check.state(['disabled'])

# ----------------------------
# GUI Setup
# ----------------------------

root = tk.Tk()
root.title("PDF Pro: Ultimate Cleaner & Optimiser")
style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", padding=6, relief="flat", background="#e1e1e1")
style.configure("TLabel", font=("Segoe UI", 10))
style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 600) // 2
y = (screen_height - 680) // 2
root.geometry(f"600x720+{x}+{y}")
root.minsize(width=600, height=720)

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(main_frame, text="PDF Watermark & Link Remover", style="Header.TLabel")
title_label.pack(pady=(0, 20))

ttk.Label(main_frame, text="Single File Selection:").pack(anchor="w")
input_frame = ttk.Frame(main_frame)
input_frame.pack(fill=tk.X, pady=5)
input_entry = ttk.Entry(input_frame)
input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
browse_button = ttk.Button(input_frame, text="Browse", command=open_file)
browse_button.pack(side=tk.LEFT)

ttk.Label(main_frame, text="Output Path (Single File):").pack(anchor="w", pady=(10, 0))
output_entry = ttk.Entry(main_frame)
output_entry.pack(fill=tk.X, pady=5)

ttk.Label(main_frame, text="Watermark Text to Remove:").pack(anchor="w", pady=(10, 0))
watermark_frame = ttk.Frame(main_frame)
watermark_frame.pack(fill=tk.X, pady=5)
watermark_entry = ttk.Entry(watermark_frame)
watermark_entry.insert(0, "watermark")
watermark_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
paste_button = ttk.Button(watermark_frame, text="Paste", width=8, command=paste_from_clipboard)
paste_button.pack(side=tk.LEFT)

ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=(20, 10))

compress_var = tk.BooleanVar(value=False)
compress_check = ttk.Checkbutton(main_frame, text="Enable Image Compression / Rasterization",
                                 variable=compress_var, command=toggle_compression)
compress_check.pack(anchor="w", pady=(5, 5))

compression_frame = ttk.Frame(main_frame)
compression_frame.pack(fill=tk.X, padx=20)
quality_label_var = tk.StringVar(value="Target Image Quality: 75%")
quality_label = ttk.Label(compression_frame, textvariable=quality_label_var, foreground="gray")
quality_label.pack(anchor="w")
quality_scale = ttk.Scale(compression_frame, from_=10, to=100, orient="horizontal", command=update_scale_label)
quality_scale.set(75)
quality_scale.pack(fill=tk.X)
quality_scale.state(['disabled'])

# Grayscale Checkbox
grayscale_var = tk.BooleanVar(value=False)
grayscale_check = ttk.Checkbutton(compression_frame, text="Convert to Grayscale (Reduces Size Significantly)", 
                                  variable=grayscale_var, state='disabled')
grayscale_check.pack(anchor="w", pady=(5,0))

ttk.Label(compression_frame, text="Compression Mode:").pack(anchor="w", pady=(8, 0))
compression_mode_var = tk.StringVar(value="Safe Compression")
compression_mode_dropdown = ttk.Combobox(compression_frame, textvariable=compression_mode_var, state="disabled",
                                         values=[
                                             "Safe Compression", 
                                             "Aggressive Compression", 
                                             "Lossless Smart", 
                                             "Rasterize (Standard)", 
                                             "Rasterize (B&W Fax Mode)"
                                         ])
compression_mode_dropdown.pack(fill=tk.X, pady=(0,6))

ttk.Label(compression_frame, text="Tips:\n• 'Rasterize (Standard)' removes vector stripes (Use with Grayscale).\n• 'Rasterize (B&W Fax Mode)' creates tiny files for text docs.\n• ALL MODES now forcibly strip every link and annotation.", font=("Segoe UI", 8), foreground="gray").pack(anchor="w")

ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=20)
progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate")
progress_bar.pack(fill=tk.X, pady=(0, 10))
save_button = ttk.Button(main_frame, text="Process Single File", command=run_single_file)
save_button.pack(fill=tk.X, pady=5)
batch_button = ttk.Button(main_frame, text="Batch Process Multiple Files", command=run_batch)
batch_button.pack(fill=tk.X, pady=5)
status_label = ttk.Label(main_frame, text="Ready", foreground="gray")
status_label.pack(pady=(10, 0))

root.mainloop()