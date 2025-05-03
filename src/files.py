"""
This file contains function for accessing and dealing with files.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import fitz
from os import listdir
from os.path import join, split, basename, isfile
from PIL import Image
import random


def get_list_of_files_with_ext(
    path_to_folder: str,
    ext: str,
    randomise: bool = False,
    verbose: bool = True
    ) -> list:
    """
        Retrieves a list of files with a specified extension from a given folder.
        Optionally randomises the list and provides verbose output.

        Parameters:
        - path_to_folder (str): Path to the folder where files will be searched.
        - ext (str): The file extension to filter files (e.g., '.txt', '.csv').
        - randomise (bool): If True, shuffles the list of files. Default is False.
        - verbose (bool): If True, prints information about the search process. Default is True.

        Returns:
        - list: A list of full file paths that match the given extension.
    """

    list_of_files = []

    # Evaluate all files in the directory
    for file in listdir(path_to_folder):

        # Skip the hidden files
        # In linux and macOS, the hidden files start
        # with '.'
        if not file.startswith('.'):

            # Get the files with the specified extension
            if file.endswith(ext):
                full_path = join(path_to_folder, file)
                list_of_files.append(full_path)

    if verbose:
        print()
        print("Looking for " + ext + " files in folder: " + path_to_folder)
        print()
        print("Total " + ext + " files found: " + str(len(list_of_files)))

    # Shuffle the list of files captured
    if randomise:
        random.shuffle(list_of_files)

    return list_of_files

def load_image_to_pil(
    path_to_image: str
    ) -> tuple:
    """
        Loads an image from a given path, converts it to RGB format, and returns the image along with its name.

        Parameters:
            path_to_image (str): The file path of the image to be loaded.

        Returns:
            tuple: A tuple containing the following:
                - PIL.Image.Image: The loaded image in RGB format.
                - str: The name of the image file.
    """

    # Get the image name
    dir, image_name = split(path_to_image)

    # Read the image
    image = Image.open(path_to_image).convert('RGB')

    # Read the image and convert it to RGB format
    return image, image_name


class PDFLoader():
    """
    This class will loads PDF file from the disc into a Python PDF utility and provide
    functions to convert the individual pages of the PDF file to images.

    Parameters:
        path_to_pdf:
            This is the path to the PDF file that needs to be loaded
        verbose:
            If this flag is set to True, the entity will produce informative
            outputs on the command line.
    """
    def __init__(
        self,
        path_to_pdf: str = None,
        verbose: bool = False
        ) -> None:
        
        # Setup the global variables
        self.verbose = verbose
        self.path_to_pdf = None
        self.pdf_name = None
        self.fitz_doc = None
        self.read_error = False
        self.page_count = None
        self.is_text_based = None

        # If path to pdf is specified in the constructor
        if path_to_pdf is not None:
            self.load_pdf(
                path_to_pdf=path_to_pdf
            )

    
    def _load_pdf_to_fitz(
        self,
        path_to_pdf: str
        ) -> tuple:
        """
        This function will load a PDF file from the filesystem
        using the PyMuPDF library into a fitz object.

        Parameters:
            path_to_pdf:
                This is the path to the PDF that needs to be
                loaded.
        """

        # Get the name of the PDF document
        _, pdf_name = split(path_to_pdf)

        # Read the PDF file into a fitz document and return it
        try:
            doc = fitz.open(path_to_pdf)

        except fitz.FileDataError:
            fitz_doc = None

        except:
            fitz_doc = None

        else:
            fitz_doc = doc
            
        return fitz_doc, pdf_name


    def _is_text_based(
        self
        ) -> None:
        """
        This  funtion will classify the PDF either as
        text-based PDF or an image-based PDF.
        """

        # Start with the assumption that the PDF is not
        # text-based
        self.is_text_based = False

        # Go through page by page and if text is found
        # then change the initial assumption
        for page in self.fitz_doc:
            
            if page.get_text("text"):

                # If the text is found, change assumption    
                self.is_text_based = True

                # Break the loop because we dont need to continue
                break
        return None


    # External Functions

    def load_pdf(
        self,
        path_to_pdf: str
        ) -> None:
        """
        This function will load the PDF file into the object of this
        class.

        Parameters:
            path_to_pdf:
                This is the path to the PDF that will be loaded.
        """

        self.path_to_pdf = path_to_pdf

        # Load the PDF and the name of the PDF
        self.fitz_doc, self.pdf_name = self._load_pdf_to_fitz(
            path_to_pdf=path_to_pdf
        )

        # Do not proceed if there was an error in reading the PDF file
        if self.fitz_doc is None:

            if self.verbose:
                print("Error Reading PDF File: " + self.pdf_name)
            
            self.read_error = True
            return None


        # Get the page count of the PDF
        self.page_count = self.fitz_doc.page_count

        # Determine if the PDF is image based or text-based
        self._is_text_based()

        # Print information about the PDF if the read is successful and the
        # verbose flag is set
        if self.verbose:
            print()
            print("Loaded File: " + self.pdf_name)
            print("Page Count of File: " + str(self.page_count))

            # Print whether text-based or not
            if self.is_text_based:
                print("PDF is text-based")
            else:
                print("PDF does not contain text")
            print()

        return None

    def get_page_in_pil(
        self,
        pg_no: int,
        dpi: int = 600
        ) -> tuple:
        """
        This function will get the page from the document that is specified
        and convert it into a PIL Image and return that.

        Parameters:
            pg_no:
                This the page number that needs to be acquired.
            dpi:
                This the DPI at which to render the image.

        Returns:
            Tuple:
                scale_factor:
                    This is the ratio of the DPI at which the PDF was rendered
                    and the original DPI setting of the PDF.

                Image:
                    This is the rendered image of the current page as a PIL
                    object.
        """

        if pg_no > self.page_count or pg_no < 0:
            print("Page number out of bounds for the PDF document")
            return None

        # Load the page
        page = self.fitz_doc.load_page(pg_no)

        _ = page.mediabox.x1
        page_height = page.mediabox.y1

        # Convert to an image
        pix = page.get_pixmap(dpi=dpi)

        # Convert the image to PIL object and return
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        _ = pix.width
        image_height = pix.height

        scale_factor = image_height/page_height

        return scale_factor, pil_image

    def transform_bb_to_pdf_space(
        self,
        bbs: list,
        pg_no: int,
        scale_factor: int = 1
        ) -> list:
        """
        In the regular coordinate space the (0, 0) position for x and y-axis
        respectively is at the top left of the page. However, in the PDF coordinate
        space the (0, 0) position for the x and y-axis respectively is at the
        bottom-left of the page and not the top-left.

        This function will convert the bounding box coordinates from the regular
        coordinate space to the PDF coordinate space. It will also scale down
        the boxes according to the scale factor of the page of PDF provided.

        Parameters:
            bb:
                This is a list of bounding boxes that needs to be transformed.
                It is a list of list with coordinates as [x1, y1, x2, y2] where
                x1, y1 is the top-left of the box and x2, y2 is the bottom-right
                of the box in the regular coordinate system.
            
            pg_no:
                This is the page number on which the bounding box exists.
            
            scale_factor:
                This is the scale factor between the image DPI and the PDF's
                original DPI.

        Returns:
            transformed_bbs:
                This is the list of transformed coordinates of the bounding boxes.
                It is a list of list with coordinates as [x1, y1, x2, y2] where
                x1, y1 is the top-left of the box and x2, y2 is the bottom-right
                of the box in the regular coordinate system.
        """

        transformed_bbs = []

        # For all bounding boxes in the list of bounding boxes provided
        for bb in bbs:

            # Scale down the bounding box to fit the original PDF size
            scaled_bb = [x/scale_factor for x in bb]

            # Load the page and get the page height
            page = self.fitz_doc.load_page(pg_no)
            page_height = page.mediabox.y1

            # Transform along the y-axis
            transformed_box = scaled_bb
            transformed_box[1] = page_height - transformed_box[1]
            transformed_box[3] = page_height - transformed_box[3]

            transformed_bbs.append(transformed_box)

        return transformed_bbs
