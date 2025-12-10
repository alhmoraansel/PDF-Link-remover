import fitz  # PyMuPDF
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os

def remove_watermarks_and_links(input_path, output_path, watermark_text="watermark"):
    """
    Removes watermarks and links from a PDF. 
    Returns True if successful, False and error message otherwise.
    """
    try:
        document = fitz.open(input_path)
    except Exception as e:
        return False, f"Failed to open PDF: {e}"

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        
        # --- BOTTLENECK FIX 1: Batch Redactions ---
        # Old way applied redactions for every instance found (Very Slow).
        # New way adds all annotations first, then applies them once per page.
        text_instances = page.search_for(watermark_text, flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        if text_instances:
            for inst in text_instances:
                page.add_redact_annot(inst, text=" ", fill=(1, 1, 1))
            
            # Apply all redactions for this page at once
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception as e:
                print(f"Error applying redactions on page {page_num}: {e}")

        # --- BOTTLENECK FIX 2: Efficient Link Removal ---
        # Old way re-fetched links in a while loop.
        # New way iterates the list once.
        links = page.get_links()
        for link in links:
            try:
                page.delete_link(link)
            except Exception as e:
                print(f"Error deleting link: {e}")

    try:
        document.save(output_path)
        document.close()
        return True, "Success"
    except Exception as e:
        return False, f"Failed to save PDF: {e}"

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        input_entry.delete(0, ctk.END)
        input_entry.insert(0, file_path)
        # Auto-suggest output name
        folder, name = os.path.split(file_path)
        name_root, ext = os.path.splitext(name)
        output_entry.delete(0, ctk.END)
        output_entry.insert(0, os.path.join(folder, f"{name_root}_cleaned{ext}"))

def save_single_file():
    input_path = input_entry.get()
    output_path = output_entry.get()
    watermark_text = watermark_entry.get()
    
    if not input_path:
        messagebox.showwarning("Warning", "Please select an input file.")
        return
        
    if not output_path:
        output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
    
    if output_path:
        success, msg = remove_watermarks_and_links(input_path, output_path, watermark_text)
        if success:
            messagebox.showinfo("Success", f"PDF saved successfully as:\n{output_path}")
        else:
            messagebox.showerror("Error", msg)

def batch_process():
    files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")], title="Select Multiple PDFs")
    if not files:
        return

    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        return

    watermark_text = watermark_entry.get()
    success_count = 0
    errors = []

    # Simple progress indication
    save_button.configure(state="disabled", text="Processing...")
    batch_button.configure(state="disabled")
    app.update()

    for file_path in files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_cleaned{ext}"
        output_path = os.path.join(output_dir, new_filename)

        success, msg = remove_watermarks_and_links(file_path, output_path, watermark_text)
        if success:
            success_count += 1
        else:
            errors.append(f"{filename}: {msg}")

    # Reset buttons
    save_button.configure(state="normal", text="Process Single File")
    batch_button.configure(state="normal")
    
    # Report results
    result_msg = f"Processed {success_count}/{len(files)} files successfully."
    if errors:
        result_msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showwarning("Batch Complete", result_msg)
    else:
        messagebox.showinfo("Batch Complete", result_msg)

# --- GUI Setup ---
app = ctk.CTk()
app.title("PDF Cleaner Pro")

# Window sizing
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
window_width = 450
window_height = 450
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
app.geometry(f"{window_width}x{window_height}+{x}+{y}")
app.minsize(width=450, height=450)

# UI Elements
# Title Section
title_label = ctk.CTkLabel(app, text="PDF Watermark & Link Remover", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

# Input Section
input_label = ctk.CTkLabel(app, text="Single File Selection:")
input_label.pack(pady=(5, 0))

input_frame = ctk.CTkFrame(app, fg_color="transparent")
input_frame.pack(pady=5)
input_entry = ctk.CTkEntry(input_frame, width=280)
input_entry.pack(side="left", padx=5)
open_button = ctk.CTkButton(input_frame, text="Browse", width=80, command=open_file)
open_button.pack(side="left")

# Output Section
output_label = ctk.CTkLabel(app, text="Output Path (Single File):")
output_label.pack(pady=(5, 0))
output_entry = ctk.CTkEntry(app, width=370)
output_entry.pack(pady=5)

# Settings Section
watermark_label = ctk.CTkLabel(app, text="Watermark Text to Remove:")
watermark_label.pack(pady=(10, 0))
watermark_entry = ctk.CTkEntry(app, width=370)
watermark_entry.insert(0, "watermark")
watermark_entry.pack(pady=5)

# Actions Section
ctk.CTkFrame(app, height=2, fg_color="gray").pack(fill="x", padx=20, pady=15) # Separator

save_button = ctk.CTkButton(app, text="Process Single File", command=save_single_file, fg_color="green", hover_color="darkgreen")
save_button.pack(pady=5, fill="x", padx=40)

batch_button = ctk.CTkButton(app, text="Batch Process Multiple Files", command=batch_process, fg_color="#3B8ED0", hover_color="#36719F")
batch_button.pack(pady=5, fill="x", padx=40)

app.mainloop()