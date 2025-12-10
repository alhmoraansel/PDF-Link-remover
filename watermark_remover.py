IndexError: list index out of range
Exception in Tkinter callback
Traceback (most recent call last):
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Lib\tkinter\__init__.py", line 1948, in __call__
    return self.func(*args)
           ^^^^^^^^^^^^^^^^
  File "C:\Source\PDF-Link-remover\watermark_remover.py", line 102, in batch_process
    success, msg = remove_watermarks_and_links(file_path, output_path, watermark_text)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Source\PDF-Link-remover\watermark_remover.py", line 33, in remove_watermarks_and_links
    links = page.get_links()
            ^^^^^^^^^^^^^^^^
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Lib\site-packages\pymupdf\__init__.py", line 11825, in get_links
    ln = ln.next
         ^^^^^^^
  File "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Lib\site-packages\pymupdf\__init__.py", line 8578, in next
    val.xref = link_xrefs[idx + 1]
               ~~~~~~~~~~^^^^^^^^^
IndexError: list index out of range