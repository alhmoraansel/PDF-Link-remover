import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, sys
import threading
import io
import tempfile
import shutil
import fitz  # PyMuPDF
import pikepdf
from pikepdf.models.image import PdfImage
from pikepdf import Name, Dictionary, Array
from PIL import Image
import concurrent.futures
import multiprocessing
import queue

# Allow loading truncated images
Image.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Helper / Compression Utilities
# ----------------------------
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



def get_raw_stream_length(obj):
    try:
        if '/Length' in obj:
            return int(obj['/Length'])
    except: pass
    try:
        return len(obj.read_raw_bytes())
    except: pass
    return 0

def pil_from_pdfimage(obj):
    try:
        pdf_img = PdfImage(obj)
        return pdf_img.as_pil_image()
    except Exception:
        return None

def resize_image(pil_img, max_dim=None):
    if not max_dim: return pil_img
    width, height = pil_img.size
    if width <= max_dim and height <= max_dim: return pil_img
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))
    
    # OPTIMIZATION: Changed LANCZOS to BILINEAR (Much faster, negligible quality loss for docs)
    return pil_img.resize((new_width, new_height), Image.Resampling.BILINEAR)

def jpeg_bytes_from_pil(pil_img, quality, subsampling=None):
    buf = io.BytesIO()
    save_kwargs = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None: save_kwargs["subsampling"] = int(subsampling)
    pil_img.save(buf, **save_kwargs)
    return buf.getvalue()

def png_bytes_from_pil(pil_img, optimize=True):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=bool(optimize))
    return buf.getvalue()

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

# ----------------------------
# Rasterization Logic
# ----------------------------

def rasterize_and_rebuild(input_pdf_path, output_pdf_path, quality, grayscale=False, fax_mode=False, progress_queue=None):
    try:
        src_doc = fitz.open(input_pdf_path)
        out_doc = fitz.open()

        if fax_mode: zoom = 2.5 
        else:
            zoom = 1.0 + (quality / 100.0)
            if zoom > 2.0: zoom = 2.0

        mat = fitz.Matrix(zoom, zoom)
        total_pages = len(src_doc)

        for i, page in enumerate(src_doc):
            if progress_queue: progress_queue.put(('progress', i + 1, total_pages))

            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB"
            if pix.n == 1: mode = "L"
            elif pix.n == 4: mode = "CMYK"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()

            if fax_mode:
                img = img.convert('L').point(lambda x: 0 if x < 200 else 255, '1')
                img.save(buf, format="TIFF", compression="group4")
            else:
                if grayscale: img = img.convert('L')
                elif mode == 'RGB':
                    try:
                        sample = img.resize((64, 64), resample=Image.NEAREST)
                        if sample.mode == 'RGB':
                            extrema = sample.getextrema()
                            if len(extrema) == 3 and extrema[0] == extrema[1] == extrema[2]:
                                img = img.convert('L')
                    except: pass

                img.save(buf, format="JPEG", quality=int(quality), optimize=True, progressive=True, subsampling=2)

            img_bytes = buf.getvalue()
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)

        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        src_doc.close()
        out_doc.close()
        return True, "Rasterization successful"
    except Exception as e:
        return False, f"Rasterization failed: {e}"

# ----------------------------
# Main Pipeline
# ----------------------------

def process_pdf_pipeline(args):
    input_path, output_path, watermark_text, quality_val, mode, grayscale, progress_queue = args
    
    fd1, temp_repaired = tempfile.mkstemp(suffix="_repaired.pdf")
    os.close(fd1)
    fd2, temp_cleaned = tempfile.mkstemp(suffix="_cleaned.pdf")
    os.close(fd2)

    try:
        if progress_queue: progress_queue.put(('status', "Step 1/3: Repairing...", 0))

        try:
            pdf = pikepdf.open(input_path, allow_overwriting_input=True)
            pdf.save(temp_repaired, fix_metadata_version=True)
            pdf.close()
        except Exception: shutil.copy2(input_path, temp_repaired)

        if progress_queue: progress_queue.put(('status', "Step 2/3: Cleaning...", 0))

        try:
            doc = fitz.open(temp_repaired)
            doc.set_metadata({}) 
            phrases = [p.strip() for p in watermark_text.split(',')] if watermark_text and watermark_text.strip() else []

            for i, page in enumerate(doc):
                try:
                    for link in page.get_links(): page.delete_link(link)
                except: pass
                try:
                    for annot in list(page.annots()):
                        if annot.type[0] == 1: page.delete_annot(annot)
                except: pass

                if phrases:
                    try:
                        for phrase in phrases:
                            if not phrase: continue
                            hits = page.search_for(phrase)
                            for rect in hits: page.add_redact_annot(rect)
                        page.apply_redactions(images=0, graphics=0)
                    except: pass
                try: page.clean_contents()
                except: pass

            doc.save(temp_cleaned, garbage=4, deflate=True)
            doc.close()
            current_input = temp_cleaned
        except Exception: current_input = temp_repaired
        
        if progress_queue: progress_queue.put(('status', "Step 3/3: Optimizing...", 0))

        if mode.startswith('rasterize'):
            is_fax = (mode == 'rasterize_fax')
            success, msg = rasterize_and_rebuild(current_input, output_path, quality_val, grayscale, fax_mode=is_fax, progress_queue=progress_queue)
            return success, msg

        try:
            pdf = pikepdf.open(current_input, allow_overwriting_input=True)
            pdf.docinfo.clear()
            try:
                if '/Outlines' in pdf.Root: del pdf.Root['/Outlines']
            except: pass
            
            max_dim = 1500 if mode == 'aggressive' else None
            all_objects = list(pdf.objects)
            
            for idx, obj in enumerate(all_objects):
                if idx % 100 == 0 and progress_queue: # OPTIMIZATION: Update UI less frequently to save CPU
                    progress_queue.put(('status', f"Optimizing Object {idx}/{len(all_objects)}...", 0))

                try:
                    if not (isinstance(obj, pikepdf.Stream) and obj.get('/Subtype') == pikepdf.Name('/Image')): continue
                    try:
                        if '/Width' in obj and int(obj['/Width']) < 50: continue
                        if '/Height' in obj and int(obj['/Height']) < 50: continue
                    except: pass

                    try:
                        pdf_img = PdfImage(obj)
                        pil_img = pil_from_pdfimage(pdf_img)
                        if pil_img is None: continue
                    except: continue

                    original_size = get_raw_stream_length(obj)
                    if grayscale and pil_img.mode != 'L': pil_img = pil_img.convert('L')
                    if max_dim: pil_img = resize_image(pil_img, max_dim)

                    new_data = None
                    is_lossless = (mode == 'lossless-smart')
                    has_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)

                    if is_lossless:
                        png_data = png_bytes_from_pil(pil_img, optimize=True)
                        if original_size == 0 or len(png_data) < original_size:
                            new_data = png_data
                            new_filter = pikepdf.Name('/FlateDecode')
                            new_colorspace = pikepdf.Name('/DeviceRGB') if pil_img.mode == 'RGB' else pikepdf.Name('/DeviceGray')
                    else:
                        if has_alpha: pil_img = flatten_alpha(pil_img)
                        if pil_img.mode not in ('RGB', 'L'): pil_img = pil_img.convert('RGB')
                        
                        q = quality_val
                        sub = 0 if mode == 'safe' else 2
                        jpeg_data = jpeg_bytes_from_pil(pil_img, q, subsampling=sub)
                        
                        accept = False
                        if original_size == 0: accept = True
                        elif len(jpeg_data) < original_size: accept = True
                        elif mode == 'aggressive' and len(jpeg_data) < original_size * 1.1: accept = True
                        
                        if accept:
                            new_data = jpeg_data
                            new_filter = pikepdf.Name('/DCTDecode')
                            new_colorspace = pikepdf.Name('/DeviceRGB') if pil_img.mode == 'RGB' else pikepdf.Name('/DeviceGray')

                    if new_data:
                        obj.clear()
                        obj.write(new_data)
                        obj['/Type'] = pikepdf.Name('/XObject')
                        obj['/Subtype'] = pikepdf.Name('/Image')
                        obj['/Width'] = pil_img.width
                        obj['/Height'] = pil_img.height
                        obj['/ColorSpace'] = new_colorspace
                        obj['/BitsPerComponent'] = 8
                        obj['/Filter'] = new_filter
                        for k in ['/Decode', '/Mask', '/SMask', '/Matte', '/ColorKeyMask', '/DecodeParms']:
                            if k in obj:
                                try: del obj[k]
                                except: pass
                except Exception: continue

            pdf.save(output_path, object_stream_mode=pikepdf.ObjectStreamMode.generate)
            pdf.close()
            return True, "Success"
        except Exception as e:
            shutil.copy2(current_input, output_path)
            return True, f"Success (Optimization skipped: {e})"
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        for p in [temp_repaired, temp_cleaned]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass

# ----------------------------
# Threading & UI Logic
# ----------------------------

def start_processing_thread(files_to_process, output_dir=None, single_output=None):
    # Disable UI
    for widget in [save_button, batch_button, browse_button, paste_wm_button]:
        widget.state(['disabled'])
    
    # Disable output entry field
    output_entry.config(state='disabled')
    
    progress_bar['value'] = 0
    status_label.config(text="Initializing engine...", foreground="#2563eb")

    quality = int(quality_scale.get()) if compress_var.get() else 100
    grayscale = grayscale_var.get()
    wm_text = watermark_entry.get().strip()

    mode_choice = compression_mode_var.get()
    if mode_choice == "Safe Compression": mode = 'safe'
    elif mode_choice == "Aggressive Compression": mode = 'aggressive'
    elif mode_choice == "Rasterize (Standard)": mode = 'rasterize'
    elif mode_choice == "Rasterize (B&W Fax Mode)": mode = 'rasterize_fax'
    else: mode = 'lossless-smart'

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    tasks = []
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        if single_output:
            current_output = single_output
        else:
            name, ext = os.path.splitext(filename)
            current_output = os.path.join(output_dir, f"{name}_cleaned{ext}")
        
        tasks.append((file_path, current_output, wm_text, quality, mode, grayscale, progress_queue))

    def poll_queue():
        try:
            while True:
                msg = progress_queue.get_nowait()
                msg_type = msg[0]
                if msg_type == 'progress':
                    _, current, total = msg
                    if total > 0:
                        percent = (current / total) * 100
                        progress_bar['value'] = percent
                        status_label.config(text=f"Processing Page {current} of {total}")
                elif msg_type == 'status':
                    _, text, _ = msg
                    status_label.config(text=text)
        except queue.Empty: pass
            
        if save_button.instate(['disabled']):
            root.after(100, poll_queue)

    root.after(100, poll_queue)

    def run_job():
        success_count = 0
        errors = []
        total_files = len(tasks)
        completed = 0
        
        # OPTIMIZATION: Use all available cores for parallel processing
        # We cap it at 4 to prevent total system freeze on older machines
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, min(4, cpu_count))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_pdf_pipeline, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                try:
                    success, msg = future.result()
                    if success: success_count += 1
                    else: errors.append(f"{filename}: {msg}")
                except Exception as exc: errors.append(f"{filename}: Exception: {exc}")
                completed += 1
                
        root.after(0, lambda: finish_processing(success_count, total_files, errors))

    threading.Thread(target=run_job, daemon=True).start()

def finish_processing(success_count, total_count, errors):
    progress_bar['value'] = 100
    status_label.config(text="Processing Complete", foreground="#10b981")
    for widget in [save_button, batch_button, browse_button, paste_wm_button]:
        widget.state(['!disabled'])
    
    # Re-enable output entry field
    output_entry.config(state='normal')
    
    result_msg = f"Processed {success_count}/{total_count} files successfully."
    if errors:
        result_msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showwarning("Completed with Issues", result_msg)
    else:
        messagebox.showinfo("Success", result_msg)

# --- UI Action Functions ---

def open_file(event=None):
    if browse_button.instate(['disabled']): return
        
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        # 1. Input Path
        full_input_path.set(file_path)
        filename = os.path.basename(file_path)
        input_label.config(text=filename, foreground="#111827", font=("Segoe UI", 9, "bold"))
        
        # 2. Output Path (Default to same folder)
        folder = os.path.dirname(file_path)
        name_root, ext = os.path.splitext(filename)
        default_out_filename = f"{name_root}_cleaned{ext}"

        # Update output directory label
        output_dir_label.config(text=folder, foreground="#1f2937", font=("Segoe UI", 8))
        
        # Update editable output filename field
        output_filename_var.set(default_out_filename)
    return "break"

def paste_watermark_text():
    try:
        text = root.clipboard_get()
        if text:
            watermark_entry.delete(0, tk.END)
            watermark_entry.insert(0, text.strip())
            status_label.config(text="Text pasted", foreground="#10b981")
            root.after(1500, lambda: status_label.config(text="Ready", foreground="#6b7280"))
    except Exception as e:
        messagebox.showerror("Paste Error", f"Could not paste: {e}")

def run_single_file():
    input_path = full_input_path.get()
    
    if not input_path:
        messagebox.showwarning("Input Missing", "Please browse for an input PDF first.")
        return
        
    # Get the directory of the input file
    input_dir = os.path.dirname(input_path)
    # Get the user-edited output filename from the Entry field
    output_filename = output_filename_var.get()
    
    if not output_filename or not output_filename.lower().endswith('.pdf'):
        messagebox.showwarning("Invalid Output Name", "Please enter a valid output filename ending in .pdf")
        return

    # Construct the final output path
    final_output_path = os.path.join(input_dir, output_filename)
        
    start_processing_thread([input_path], single_output=final_output_path)

def run_batch():
    files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")], title="Select Multiple PDFs")
    if not files: return
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir: return
    start_processing_thread(files, output_dir=output_dir)

def update_scale_label(val):
    quality_label_var.set(f"Quality: {int(float(val))}%")

def toggle_compression():
    if compress_var.get():
        quality_scale.state(['!disabled'])
        compression_mode_dropdown.state(['!disabled'])
        compression_mode_dropdown.config(state="readonly")
        grayscale_check.state(['!disabled'])
    else:
        quality_scale.state(['disabled'])
        compression_mode_dropdown.state(['disabled'])
        grayscale_check.state(['disabled'])

# ----------------------------
# GUI Setup
# ----------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root = tk.Tk()
    root.title("CleanPDF Mini")
    root.geometry("450x480") 
    root.minsize(400, 430)
    
    # --- NEW: SET ICON ---
    # Make sure you have a file named 'app.ico' in the same folder
    try:
        root.iconbitmap(resource_path("app.ico"))
    except Exception:
        pass # Fallback if icon missing
    
    # Internal variables
    full_input_path = tk.StringVar()
    output_filename_var = tk.StringVar(value="_cleaned.pdf")
    
    style = ttk.Style()
    style.theme_use('clam') 

    BG_MAIN = "#f3f4f6"       
    BG_CARD = "#ffffff"       
    TEXT_MAIN = "#1f2937"     
    TEXT_SUB = "#6b7280"      
    ACCENT = "#2563eb"        
    ACCENT_HOVER = "#1d4ed8"  
    BORDER = "#d1d5db"        

    root.configure(bg=BG_MAIN)

    style.configure(".", background=BG_MAIN, foreground=TEXT_MAIN, font=("Segoe UI", 9))
    style.configure("Card.TFrame", background=BG_CARD, relief="flat")
    style.configure("Card.TLabelframe", background=BG_CARD, bordercolor=BORDER, relief="solid", borderwidth=1)
    style.configure("Card.TLabelframe.Label", background=BG_CARD, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
    
    style.configure("TButton", padding=(5, 3), font=("Segoe UI", 8, "bold"), background="#e5e7eb", borderwidth=0)
    style.map("TButton", background=[('active', '#d1d5db')])
    
    style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 9, "bold"), borderwidth=0)
    style.map("Accent.TButton", background=[('active', ACCENT_HOVER)])

    style.configure("Modern.TEntry", fieldbackground=BG_MAIN, bordercolor=BORDER, padding=3)
    style.configure("Horizontal.TProgressbar", thickness=10, background=ACCENT, troughcolor="#e5e7eb", bordercolor=BG_MAIN)

    main_container = ttk.Frame(root, padding=10)
    main_container.pack(fill=tk.BOTH, expand=True)

    # --- Actions (Bottom) ---
    action_frame = ttk.Frame(main_container)
    action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    status_label = ttk.Label(action_frame, text="Ready", foreground=TEXT_SUB, font=("Segoe UI", 8))
    status_label.pack(anchor="w", pady=(0, 2))

    progress_bar = ttk.Progressbar(action_frame, orient="horizontal", mode="determinate", style="Horizontal.TProgressbar")
    progress_bar.pack(fill=tk.X, pady=(0, 8))

    btn_grid = ttk.Frame(action_frame)
    btn_grid.pack(fill=tk.X)

    save_button = ttk.Button(btn_grid, text="PROCESS ONE", style="Accent.TButton", command=run_single_file)
    save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    batch_button = ttk.Button(btn_grid, text="BATCH FOLDER", style="Accent.TButton", command=run_batch)
    batch_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    # --- Content (Middle) ---
    
    # 1. File Config
    file_frame = ttk.LabelFrame(main_container, text=" Files ", padding=(10, 8), style="Card.TLabelframe")
    file_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
    file_frame.columnconfigure(1, weight=1)

    # Input 
    ttk.Label(file_frame, text="Input:", background=BG_CARD, font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w")
    
    input_label = ttk.Label(file_frame, text="No file selected", background=BG_CARD, foreground="#9ca3af", font=("Segoe UI", 9, "italic"))
    input_label.grid(row=0, column=1, sticky="w", padx=10)
    
    browse_button = ttk.Button(file_frame, text="Browse", width=6, command=open_file)
    browse_button.grid(row=0, column=2, sticky="e")

    # Output Folder (Read-only for info)
    ttk.Label(file_frame, text="Output Directory:", background=BG_CARD, font=("Segoe UI", 8, "bold")).grid(row=1, column=0, sticky="w", pady=(8,2), columnspan=3)
    
    output_dir_label = ttk.Label(file_frame, text="[Select file to set directory]", background=BG_CARD, foreground="#9ca3af", font=("Segoe UI", 8, "italic"))
    output_dir_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 8))

    # Output Filename (Editable Entry)
    ttk.Label(file_frame, text="Output Filename:", background=BG_CARD, font=("Segoe UI", 8, "bold")).grid(row=3, column=0, sticky="w")
    
    output_entry = ttk.Entry(file_frame, textvariable=output_filename_var, style="Modern.TEntry")
    output_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0))

    # Watermark
    ttk.Label(file_frame, text="Watermark Text:", background=BG_CARD, font=("Segoe UI", 8, "bold")).grid(row=4, column=0, sticky="w", pady=(8,0))
    watermark_entry = ttk.Entry(file_frame, style="Modern.TEntry")
    watermark_entry.grid(row=4, column=1, sticky="ew", padx=10, pady=(8,0))
    
    paste_wm_button = ttk.Button(file_frame, text="Paste", width=5, command=paste_watermark_text)
    paste_wm_button.grid(row=4, column=2, sticky="e", pady=(8,0))

    # 2. Settings
    opt_frame = ttk.LabelFrame(main_container, text=" Settings ", padding=(10, 8), style="Card.TLabelframe")
    opt_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

    compress_var = tk.BooleanVar(value=False)
    compress_check = ttk.Checkbutton(opt_frame, text="Enable Advanced Compression", variable=compress_var, command=toggle_compression, style="Switch.TCheckbutton")
    compress_check.pack(anchor="w", pady=(0, 5))
    
    ttk.Separator(opt_frame, orient='horizontal').pack(fill='x', pady=(2, 8))

    settings_inner = ttk.Frame(opt_frame, style="Card.TFrame")
    settings_inner.pack(fill=tk.X)

    ttk.Label(settings_inner, text="Mode:", background=BG_CARD, foreground=TEXT_SUB, font=("Segoe UI", 8)).pack(anchor="w")
    compression_mode_var = tk.StringVar(value="Safe Compression")
    compression_mode_dropdown = ttk.Combobox(settings_inner, textvariable=compression_mode_var, state="disabled",
                                             values=[
                                                 "Safe Compression", 
                                                 "Aggressive Compression", 
                                                 "Lossless Smart",
                                                 "Rasterize (Standard)",
                                                 "Rasterize (B&W Fax Mode)"
                                             ], font=("Segoe UI", 9))
    compression_mode_dropdown.pack(fill=tk.X, pady=(0, 8))

    quality_label_var = tk.StringVar(value="Quality: 75%")
    ttk.Label(settings_inner, textvariable=quality_label_var, background=BG_CARD, foreground=TEXT_SUB, font=("Segoe UI", 8)).pack(anchor="w")
    
    quality_scale = ttk.Scale(settings_inner, from_=10, to=100, orient="horizontal", command=update_scale_label)
    quality_scale.set(75)
    quality_scale.pack(fill=tk.X, pady=(0, 8))
    quality_scale.state(['disabled'])

    grayscale_var = tk.BooleanVar(value=False)
    grayscale_check = ttk.Checkbutton(settings_inner, text="Grayscale", variable=grayscale_var, state='disabled')
    grayscale_check.pack(anchor="w")

    root.mainloop()