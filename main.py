import os
import io
import shutil
import tempfile
import threading
import concurrent.futures
import flet as ft
import fitz  # PyMuPDF
from PIL import Image

# ----------------------------
# Backend Logic (PyMuPDF Only - Termux Compatible)
# ----------------------------

def rasterize_page(page, quality, grayscale, fax_mode):
    """Converts a PDF page to an image and returns the image bytes."""
    # Zoom factor for quality
    zoom = 2.0 if fax_mode else (1.0 + (quality / 100.0))
    mat = fitz.Matrix(zoom, zoom)
    
    # Render page to pixels
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to PIL for processing
    mode = "RGB"
    if pix.n == 1: mode = "L"
    elif pix.n == 4: mode = "CMYK"
    
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()

    if fax_mode:
        # B&W Fax Style (Thresholding)
        img = img.convert('L').point(lambda x: 0 if x < 200 else 255, '1')
        img.save(buf, format="PNG", optimize=True) # PNG is safer for B&W text
    else:
        # Standard JPEG compression
        if grayscale: img = img.convert('L')
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    
    return buf.getvalue(), img.size

def process_pdf_pipeline(args):
    # args: (input_path, output_path, watermark_text, quality_val, mode, grayscale, progress_callback)
    input_path, output_path, watermark_text, quality_val, mode, grayscale, progress_callback = args
    
    temp_output = tempfile.mktemp(suffix=".pdf")
    
    try:
        if progress_callback: progress_callback('status', "Opening PDF...", 0)
        
        doc = fitz.open(input_path)
        total_pages = len(doc)
        
        # --- PHASE 1: Cleaning (Links & Metadata) ---
        if progress_callback: progress_callback('status', "Removing Links & Metadata...", 0)
        
        doc.set_metadata({
            'producer': 'CleanPDF',
            'creator': 'CleanPDF',
            'title': '',
            'author': ''
        })

        phrases = [p.strip() for p in watermark_text.split(',')] if watermark_text.strip() else []

        for i, page in enumerate(doc):
            if progress_callback: 
                progress_callback('progress', i + 1, total_pages)
            
            # 1. Remove Links
            clean_links_count = 0
            for link in page.get_links():
                page.delete_link(link)
                clean_links_count += 1
            
            # 2. Remove Annotations (Popups, Highlights)
            for annot in list(page.annots()):
                page.delete_annot(annot)
                
            # 3. Text Watermark Redaction (Basic)
            if phrases:
                for phrase in phrases:
                    if not phrase: continue
                    hits = page.search_for(phrase)
                    for rect in hits:
                        page.add_redact_annot(rect)
                page.apply_redactions(images=0, graphics=0)

        # --- PHASE 2: Rasterization (If Selected) ---
        if mode.startswith('rasterize'):
            out_doc = fitz.open()
            is_fax = (mode == 'rasterize_fax')
            
            for i, page in enumerate(doc):
                if progress_callback:
                    progress_callback('status', f"Rasterizing Page {i+1}/{total_pages}...", 0)
                
                # Render page to image
                img_bytes, dimensions = rasterize_page(page, quality_val, grayscale, is_fax)
                
                # Create new page in output doc with same dimensions
                # Note: dimensions from PIL are (width, height)
                new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                
                # Insert the flattened image
                new_page.insert_image(new_page.rect, stream=img_bytes)
            
            # Save the rasterized doc
            out_doc.save(output_path, garbage=4, deflate=True)
            out_doc.close()
            doc.close()
            return True, "Rasterization Successful"

        # --- PHASE 3: Standard Optimization ---
        # If not rasterizing, just save the cleaned doc with garbage collection
        if progress_callback: progress_callback('status', "Saving & Compressing...", 0)
        
        # 'garbage=4' is aggressive deduplication
        # 'deflate=True' compresses streams
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()

        return True, "Success"

    except Exception as e:
        return False, f"Error: {e}"

# ----------------------------
# UI / Flet Application
# ----------------------------

def main(page: ft.Page):
    page.title = "CleanPDF Mini (Termux)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.padding = 20

    # --- State Variables ---
    selected_files = []
    output_dir = None
    
    # --- UI Elements ---
    status_text = ft.Text("Ready", size=12, color=ft.colors.GREY_700)
    progress_bar = ft.ProgressBar(value=0, width=400, color=ft.colors.BLUE, visible=False)
    
    input_file_label = ft.Text("No file selected", italic=True, color=ft.colors.GREY_500)
    output_filename_field = ft.TextField(label="Output Filename", value="_cleaned.pdf", text_size=12, dense=True)
    watermark_field = ft.TextField(label="Watermark Text", hint_text="Text to redact", text_size=12, dense=True)
    
    # Settings Elements
    enable_compress_switch = ft.Switch(label="Enable Advanced Mode", value=False)
    
    mode_dropdown = ft.Dropdown(
        label="Processing Mode",
        options=[
            ft.dropdown.Option("Clean Links & Optimize"),
            ft.dropdown.Option("Rasterize (Standard)"),
            ft.dropdown.Option("Rasterize (B&W Fax Mode)"),
        ],
        value="Clean Links & Optimize",
        disabled=True,
        dense=True
    )
    
    quality_slider = ft.Slider(min=10, max=100, divisions=90, value=75, label="{value}%", disabled=True)
    quality_label = ft.Text("Quality: 75%", color=ft.colors.GREY)
    grayscale_switch = ft.Switch(label="Grayscale", value=False, disabled=True)

    # --- Event Handlers ---

    def on_compress_change(e):
        enabled = enable_compress_switch.value
        mode_dropdown.disabled = not enabled
        quality_slider.disabled = not enabled
        grayscale_switch.disabled = not enabled
        page.update()

    def on_slider_change(e):
        quality_label.value = f"Quality: {int(quality_slider.value)}%"
        page.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        nonlocal selected_files, output_dir
        if e.files:
            selected_files = e.files
            first_file = selected_files[0]
            input_file_label.value = f"{first_file.name} (+{len(selected_files)-1} others)" if len(selected_files) > 1 else first_file.name
            input_file_label.color = ft.colors.BLACK
            input_file_label.italic = False
            
            name, ext = os.path.splitext(first_file.name)
            output_filename_field.value = f"{name}_cleaned{ext}"
            
            try:
                output_dir = os.path.dirname(first_file.path)
            except:
                output_dir = None
            
            page.update()

    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)

    def update_progress_ui(msg_type, val1, val2):
        if msg_type == 'progress':
            if val2 > 0:
                progress_bar.value = val1 / val2
                status_text.value = f"Processing Page {val1} of {val2}"
        elif msg_type == 'status':
            status_text.value = val1
        page.update()

    def run_processing():
        if not selected_files:
            page.show_snack_bar(ft.SnackBar(ft.Text("Please select files first")))
            return

        process_btn.disabled = True
        batch_btn.disabled = True
        progress_bar.visible = True
        progress_bar.value = None
        page.update()

        # Gather Settings
        quality = int(quality_slider.value)
        grayscale = grayscale_switch.value
        wm_text = watermark_field.value.strip()
        
        mode_choice = mode_dropdown.value
        if not enable_compress_switch.value:
            mode = 'safe'
        elif mode_choice == "Rasterize (Standard)": 
            mode = 'rasterize'
        elif mode_choice == "Rasterize (B&W Fax Mode)": 
            mode = 'rasterize_fax'
        else: 
            mode = 'safe'

        # Output logic
        target_dir = output_dir if output_dir else os.getcwd()

        tasks = []
        for f in selected_files:
            inp_path = f.path
            
            if len(selected_files) == 1:
                out_name = output_filename_field.value
            else:
                name, ext = os.path.splitext(f.name)
                out_name = f"{name}_cleaned{ext}"
            
            out_path = os.path.join(target_dir, out_name)
            tasks.append((inp_path, out_path, wm_text, quality, mode, grayscale))

        success_count = 0
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {}
            for task in tasks:
                task_with_cb = task + (update_progress_ui,)
                future = executor.submit(process_pdf_pipeline, task_with_cb)
                future_to_file[future] = task[0]

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, msg = future.result()
                    if success: success_count += 1
                    else: errors.append(f"{os.path.basename(file_path)}: {msg}")
                except Exception as exc:
                    errors.append(f"{os.path.basename(file_path)}: {exc}")

        progress_bar.value = 0
        progress_bar.visible = False
        process_btn.disabled = False
        batch_btn.disabled = False
        
        final_msg = f"Processed {success_count}/{len(tasks)} files."
        
        if errors:
            status_text.value = "Errors occurred."
            status_text.color = ft.colors.ORANGE
            page.show_snack_bar(ft.SnackBar(ft.Text(f"Error: {errors[0]}")))
        else:
            status_text.value = "Success! Saved to " + target_dir
            status_text.color = ft.colors.GREEN
            page.show_snack_bar(ft.SnackBar(ft.Text(final_msg)))
        
        page.update()

    process_btn = ft.ElevatedButton(
        "Process PDF", 
        icon=ft.icons.PLAY_ARROW, 
        on_click=lambda e: run_processing(), 
        bgcolor=ft.colors.BLUE, 
        color=ft.colors.WHITE,
        width=200
    )
    
    batch_btn = ft.ElevatedButton(
        "Select Files", 
        icon=ft.icons.FOLDER_OPEN, 
        on_click=lambda _: file_picker.pick_files(allow_multiple=True, allowed_extensions=["pdf"]),
        width=200
    )

    # --- Layout ---
    page.add(
        ft.Card(
            content=ft.Container(
                padding=15,
                content=ft.Column([
                    ft.Text("CleanPDF Mini", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text("Termux Edition", size=12, color=ft.colors.GREY),
                    ft.Divider(),
                    ft.Text("1. Select PDF", weight=ft.FontWeight.BOLD),
                    ft.Row([input_file_label]),
                    ft.Container(batch_btn, padding=5),
                    output_filename_field,
                    watermark_field
                ])
            )
        ),
        ft.Card(
            content=ft.Container(
                padding=15,
                content=ft.Column([
                    ft.Text("2. Settings", weight=ft.FontWeight.BOLD),
                    enable_compress_switch,
                    mode_dropdown,
                    quality_label,
                    quality_slider,
                    grayscale_switch
                ])
            )
        ),
        ft.Divider(),
        status_text,
        progress_bar,
        ft.Container(process_btn, alignment=ft.alignment.center, padding=10)
    )

ft.app(target=main, view=ft.AppView.WEB_BROWSER)