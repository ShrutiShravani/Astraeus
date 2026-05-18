import os
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
import shutil
from dotenv import load_dotenv
import asyncio
from pypdf import PdfReader, PdfWriter
import gc 

load_dotenv()

semaphore = asyncio.Semaphore(1)

BASE_DIR = Path(os.getenv("DATA_DIR"))

INPUT_DIR = BASE_DIR / "raw"

OUTPUT_DIR = BASE_DIR / "extracted_md"

DLQ_DIR = BASE_DIR / "dead_letter_queue"

TEMP_DIR = BASE_DIR / "temp_chunks"


for path in [INPUT_DIR, OUTPUT_DIR, DLQ_DIR, TEMP_DIR]:

    path.mkdir(parents=True, exist_ok=True)


def split_pdf(input_path, temp_dir, chunk_size=8):

    temp_job_dir = temp_dir / f"tmp_{input_path.stem}"

    temp_job_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(input_path)

    total_pages = len(reader.pages)

    chunks = []


    for i in range(0, total_pages, chunk_size):

        writer = PdfWriter()

        end_page = min(i + chunk_size, total_pages)


        for page_num in range(i, end_page):

            writer.add_page(reader.pages[page_num])


        chunk_filename = f"{input_path.stem}_start_{i+1}.pdf"

        chunk_path = temp_job_dir / chunk_filename


        with open(chunk_path, "wb") as f:

            writer.write(f)


        chunks.append((chunk_path, i))

    return chunks, temp_job_dir

async def process_chunks(converter, chunk_path, page_offset, chunk_idx, total, pdf_name):
    """Processes a single chunk with full precision in parallel."""
    async with semaphore:
        print(f"[{pdf_name}] Chunk {chunk_idx}")
        try:
            result = await asyncio.to_thread(converter.convert,chunk_path)
            full_doc_md = []
            doc = result.document

            last_page_seen = -1

            for element, _ in doc.iterate_items():

                if getattr(element, "prov", None):

                    current_page = element.prov[0].page_no + 1 + page_offset

                    if current_page != last_page_seen:

                        full_doc_md.append(f"\n\n## PAGE {current_page}\n")

                        full_doc_md.append("=" * 20 + "\n\n")

                        last_page_seen = current_page

                try:

                    full_doc_md.append(element.export_to_markdown() + "\n")

                except Exception:

                    if hasattr(element, "text"):

                        full_doc_md.append(element.text + "\n")


            gc.collect()
            return page_offset,"".join(full_doc_md)
        except Exception as e:
            print(f"Error in chunk {chunk_idx}: {e}")
            raise


async def extract_final_v3(pdf_files: list[Path]):

    pdf_options = PdfPipelineOptions()

    pdf_options.do_table_structure = True

    pdf_options.table_structure_options.mode = TableFormerMode.FAST

    pdf_options.do_ocr = False

    full_md=[]
    results=[]

    converter = DocumentConverter(

        format_options={

            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)

        }

    )


    for pdf_path in pdf_files:

        temp_job_dir = None
 
        try:

            chunks, temp_job_dir = split_pdf(pdf_path, TEMP_DIR, chunk_size=3)


            subfolder = pdf_path.parent.name if pdf_path.parent != INPUT_DIR else ""

            target_dir = OUTPUT_DIR / subfolder


            target_dir.mkdir(parents=True, exist_ok=True)


            output_file = target_dir / f"{pdf_path.stem}.md"

            with open(output_file,"w",encoding="utf-8") as f:
                for i,(cp,offset) in enumerate(chunks):
                    _,text=await(process_chunks(
                        converter,
                        cp,
                        offset,
                        i + 1,
                        len(chunks),
                        pdf_path.name
                    ))


                    f.write(text)

                    del text
                    gc.collect()


            results.append(output_file)

            pdf_success = True

            print(f"FINAL SUCCESS: {pdf_path.name}")


        except Exception as e:

            print(f"PDF attempt failed for {pdf_path.name}: {e}")


            shutil.move(str(pdf_path), str(DLQ_DIR / pdf_path.name))

            print(f"Moved to DLQ: {pdf_path.name}")


        finally:

            if temp_job_dir is not None and temp_job_dir.exists():

                shutil.rmtree(temp_job_dir, ignore_errors=True)



    return results


if __name__ == "__main__":

    files = list(INPUT_DIR.rglob("*.pdf"))

    asyncio.run(extract_final_v3(files))