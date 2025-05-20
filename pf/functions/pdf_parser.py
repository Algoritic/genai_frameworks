import concurrent.futures
import os
import tempfile
from functools import partial

import magic
import pymupdf
from logger import logger
from promptflow.tracing import trace


@trace
def get_file_mimetype(file: bytes) -> str | None:
    return magic.from_buffer(file, mime=True)


@trace
def process_page(page, output_folder, file_name):
    """Process a single PDF page and save it as an image."""
    try:
        logger.info(f"Processing page {page.number}")
        pix = page.get_pixmap(dpi=400)
        output_path = f"{output_folder}/{file_name}-page-{page.number}.png"
        pix.save(output_path)
        logger.info(f"Saved page {page.number} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error processing page {page.number}: {str(e)}")
        return None


@trace
def process_page_bytes(page=None, doc=None, extract_image=False):
    """Process a single PDF page and return it as bytes."""
    try:
        logger.info(f"Processing page {page.number}")
        if extract_image:
            images = page.get_images(full=True)
            if not images:
                logger.info("No images found on page %i" % page.number)
                return None
            for _, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                return image_bytes  # return the first image found

        pix = page.get_pixmap(dpi=400)
        logger.info(f"Processed page {page.number}")
        return pix.tobytes()
    except Exception as e:
        logger.error(f"Error processing page {page.number}: {str(e)}")
        return None


@trace
def bytes_pdf_to_image_bytes(file: bytes, extract_image=False) -> list:
    doc = pymupdf.open(stream=file)
    page_count = doc.page_count
    ext = doc.metadata.get("format", "")

    logger.info(f"Total pages: {page_count}")
    logger.info(f"File type: {ext}")
    assert "PDF" in ext, "File is not a PDF"

    # Determine optimal number of workers based on CPU count and page count
    max_workers = min(os.cpu_count() or 4, page_count)

    results = [None] * page_count  # Preallocate list for ordered results

    def task(index):
        try:
            return index, process_page_bytes(
                doc[index], doc=doc, extract_image=extract_image
            )
        except Exception as e:
            logger.error(f"Error processing page {index}: {e}")
            return index, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, i) for i in range(page_count)]

        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result

    # Filter out None results (in case of page processing failures)
    results = [r for r in results if r is not None]

    logger.info(f"Processed {len(results)} pages")

    return results


@trace
def pdf_to_images(
    file: bytes,
    file_ext: str,
    use_in_memory: bool = False,
    first_page_only: bool = False,
    extract_image=True,
) -> str:
    """
    Convert a PDF file to images using parallel processing.

    Args:
        file: The file content as bytes
        file_ext: The file extension (e.g., ".pdf")
        first_page_only: If True, only convert the first page
        extract_image: If True, extract images from the PDF

    Returns:
        The path to the directory containing the output images
    """
    file_type = get_file_mimetype(file)
    logger.info(f"File type: {file_type}")

    # Create output directory
    output_folder = tempfile.mkdtemp()

    # Write the temporary file
    with tempfile.NamedTemporaryFile(
        dir=output_folder if file_ext != ".pdf" else None,
        suffix=file_ext,
        delete=use_in_memory,
    ) as temp_file:
        temp_file.write(file)
        temp_file.seek(0)
        file_name = os.path.basename(temp_file.name)

        if file_type == "application/pdf":
            logger.info("Converting PDF to images")
            doc = pymupdf.open(temp_file.name)
            page_count = len(doc)
            logger.info(f"Total pages: {page_count}")

            if extract_image:

                def extract_images_from_page(page):
                    images = page.get_images(full=True)
                    if not images:
                        logger.info("No images found on page %i" % page.number)
                        return []
                    extracted = []
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_name = f"{output_folder}/{file_name}-page-{page.number}-{img_index}.{image_ext}"
                        with open(image_name, "wb") as img_file:
                            img_file.write(image_bytes)
                        extracted.append(image_name)
                    return extracted

                max_workers = min(os.cpu_count() or 4, len(doc))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(extract_images_from_page, doc[i])
                        for i in range(len(doc))
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        # Just to raise exceptions if any
                        future.result()

                logger.info("Extracted images from PDF")
                return output_folder

            # Handle first page only case
            if first_page_only:
                process_page(doc[0], output_folder, file_name)
                return output_folder

            # Process pages in parallel
            process_func = partial(
                process_page, output_folder=output_folder, file_name=file_name
            )

            # Determine optimal number of workers based on CPU count and page count
            max_workers = min(os.cpu_count() or 4, page_count)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all pages for processing
                futures = [
                    executor.submit(process_func, doc[i]) for i in range(page_count)
                ]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

                # Check for any errors
                for future in futures:
                    if future.exception():
                        logger.error(f"Page processing error: {future.exception()}")
        else:
            logger.info("Image received, no conversion needed")

        return output_folder
