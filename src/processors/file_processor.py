from core.logger import logger
import os
import tempfile
import pymupdf
import magic
import cv2


def get_file_mimetype(file: bytes) -> str | None:
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file)
        temp_file.seek(0)
        logger.info("Guessing file type for %s" % temp_file.name)
        return magic.from_buffer(temp_file.read(), mime=True)


def batch_optimize_image(folder_path: str) -> str:  #return folder path
    #load all files under the folder
    files = os.listdir(folder_path)
    if len(files) == 0:
        raise ValueError("No files found in folder")
    for f in files:
        image = cv2.imread(os.path.join(folder_path, f))
        if (image is None):
            raise ValueError("Unable to open image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        cv2.imwrite(os.path.join(folder_path, f), denoised)
    return folder_path


def pdf_to_images(file: bytes, file_ext: str) -> str:  #dir path
    file_type = get_file_mimetype(file)
    logger.info(file_type)
    output_folder = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile(
            dir=output_folder if file_ext != ".pdf" else None,
            suffix=file_ext,
            delete=False,
    ) as temp_file:
        temp_file.write(file)
        temp_file.seek(0)
        file_name = os.path.basename(temp_file.name)
        if file_type == "application/pdf":
            logger.info("Converting pdf to images")
            doc = pymupdf.open(temp_file.name)
            logger.info("total pages %s" % len(doc))
            for page in doc:
                logger.info("Saving %s" % output_folder)
                pix = page.get_pixmap(dpi=300)
                pix.save(f"{output_folder}/{file_name}-page-%i.png" %
                         page.number)
        else:
            logger.info("Image received")
            # temp_file.write(file)
            # temp_file.seek(0)
            return output_folder
    return output_folder
