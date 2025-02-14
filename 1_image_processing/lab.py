#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math
import os
from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col):
    """
    returns the pixel requested at row, col
    """
    return image["pixels"][row][col]
    # return image["pixels"][col, row]


def set_pixel(image, row, col, color):
    """
    sets the specific pixel to a new color passed in
    """
    image["pixels"][row][col] = color

def apply_per_pixel(image, func):
    """
    makes a new result but loops over the pixels and applies func to it
    """
    result = {
        "height": image["height"],
        "width": image["width"], # fixed typo
        "pixels": image["pixels"].copy(),
    }
    for row in range(image["height"]): # switched names to row
        for col in range(image["width"]):
            # gets each pixel and finds the new color from func, then sets it as result
            color = get_pixel(image, row, col) # switched order row and col
            new_color = func(color)
            set_pixel(result, row, col, new_color) # tabbed this in one

    # original:
    # for col in range(image["height"]):
    #     for row in range(image["width"]):
    #         color = get_pixel(image, col, row)
    #         new_color = func(color)
    #     set_pixel(result, row, col, new_color)
    return result


def fix_list(image):
    """
    takes image and returns a new pixels array, 
    but this time it's 2d, with many lists of rows as inner lists.
    This way you can get a specific pixel much easier, by knowing the row 
    and column
    input: 
    {
    "height": 3,
    "width": 2,
    "pixels": [0, 50, 50, 100, 100, 255],
    }

    output: [[0, 50], [50, 100], [100, 255]]
    """
    final = []
    for i in range(len(image["pixels"])):
        if i % image["width"] == 0:
            final.append([image["pixels"][i]])
        else:
            final[-1].append(image["pixels"][i])
    return final
        # image["height"]

def inverted(image):
    """
    makes a new image dictionary but with a fixed_list from the function
    which makes it row and column indexable.
    """
    # turning 1d array into 2d array with rows and cols
    new_pixels = fix_list(image)
    new_image = {
    "height": image["height"],
    "width": image["width"],
    "pixels": new_pixels,
}
    # calling apply_per_pixel on this new image with 2d arraw pixels
    inverted_image = apply_per_pixel(new_image, lambda color: 255-color)
    # changed from 256-color

    final = []
    for i in range(len(inverted_image["pixels"])):
        # turning 2d array back into 1d for submission
        for j in inverted_image["pixels"][i]:
            final.append(j)
    inverted_image["pixels"] = final
    return inverted_image


# HELPER FUNCTIONS
def get_pixel_bounds(image, row, col, behavior):
    """
    returns the pixel requested at row, col, 
    with options for out of bounds behavior

    this is a complicated function due to the out of bound behavior
    there is an if statement for every possible out of bound
    if row, col is
    + +, + -, - -, - +
    """

    # If everything is within bounds
    if 0 <= row < image["height"] and 0 <= col < image["width"]:
        # get the pixel, however, it's not a 2d array so we use math to get it
        return image["pixels"][image["width"]*row + col]

    # if pixel row and col is -
    if col < 0 and row < 0:
        if behavior == "zero":
            return 0
        if behavior == "extend":
            # get the first pixel
            return image["pixels"][0]
        elif behavior == "wrap":
            # return the pixel of row % height * width + col % width,
            # acting like a repeat wrap
            # out of bounds diagonally top left turns into pixel
            # from the bottom right
            return image["pixels"][row % image["height"]*image["width"] + col % image["width"]]

    # if pixel row and col are both out of bounds +
    if col > image["width"]-1 and row > image["height"]-1:
        if behavior == "extend":
            # get the last pixel
            return image["pixels"][-1]
        elif behavior == "wrap":
            # return the pixel of row % height * height + col % width,
            # acting like a repeat wrap
            # out of bounds diagonally bottom right turns into pixel from the top left


            return image["pixels"][(row % image["height"]) * image["width"] + (col % image["width"])]
    # if pixel row and col are both out of bounds, row -, col +
    if col > image["width"]-1 and row < 0:
        if behavior == "extend":
            # get the last pixel in the first row
            return image["pixels"][image["width"] - 1]
        elif behavior == "wrap":
            # return the pixel of row % height * height + col % width,
            # acting like a repeat wrap
            # out of bounds diagonally bottom right turns into
            # pixel from the top left

            return image["pixels"][(row % image["height"]) * image["width"] + (col % image["width"])]


    # if pixel row and col are both out of bounds, row +, col -
    if col < 0 and row > image["height"]-1:
        if behavior == "extend":
            # get the first pixel in last row
            return image["pixels"][-image["width"]]
        elif behavior == "wrap":
            # return the pixel of row % height * height + col % width,
            # acting like a repeat wrap
            # out of bounds diagonally bottom right turns into pixel
            # from the top left

            return image["pixels"][(row % image["height"]) * image["width"] + (col % image["width"])]


    # if the row is out of bounds, pixel is vertically under or above the picture
    if row >= image["height"] or row < 0:
        if behavior == "zero":
            return 0
        elif behavior == "extend":
            # if row is -, get the first pixel in frame in that column
            if row < 0:
                return image["pixels"][col]

            # gets the last pixel in the frame in the same column
            else:
                return image["pixels"][image["width"]*(image["height"]-1)+col]
        elif behavior == "wrap":
            # return the pixel of row % height row + col, acting like a repeat wrap
            # if row was 5, height=3, it would take 5 % 3 = 2, so it would
            # actually take the 2nd row
            return image["pixels"][(row % image["height"])*image["width"] + col]

    # if the column is out of bounds, pixel is to the left or right of picture
    if col >= image["width"] or col < 0:
        if behavior == "zero":
            return 0
        elif behavior == "extend":
            # gets the last or first pixel in the frame in the same row
            if col < 0:
                # if col is -, get the pixel at position width*row.
                # The 0th element on that row
                return image["pixels"][image["width"]*row]
            # else, get the pixel at position width*row+(width-1).
            # The last element on that row
            return image["pixels"][image["width"]*row+(image["width"]-1)]
        elif behavior == "wrap":
            # return the pixel of row % height, acting like a repeat wrap
            # if row was 5, height=3, it would take 5 % 3 = 2, so it would
            # actually take the 2nd row
            return image["pixels"][col % image["width"] + row*image["width"]]


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Kernel:
    a 2d array, for ex:
    [[0,1,0],[0,0,0],[0,0,0]]
    [0,1,0,0,0,0,0,0,0]
    """

    # if no valid boundary_behavior, return None
    if boundary_behavior != "zero" and boundary_behavior != "extend" and boundary_behavior != "wrap":
        return None

    # making a temp duplicate array
    final_pixels = image["pixels"].copy()

    # loops over every current pixel
    for i in range(len(final_pixels)):
        new_pixel=0 # initializes new_pixel to 0, but will add to it

        # int(len(kernel)/2) finds how many to go in the negative dir.
        # for 3x3, it's -1 -> 1. for 5x5, it's -2 -> 2
        # then loops from that negative number to its abs value + 1.
        # Sandwiches the number we are calculating
        # these next two lines are making a box, applying the kernel.
        # so if the kernel is 3x3, it will check the elements
        # all around and including the ith element

        for j in range(-int(len(kernel)/2), int(len(kernel)/2)+1):
            for k in range(-int(len(kernel)/2), int(len(kernel)/2)+1):
                # gets the value of this specific pixel.
                # It is relative to the ith pixel, in a box around it
                val = get_pixel_bounds(image,int(i / image["width"])+j,(i % image["width"])+k, boundary_behavior)
                # with this pixel, multiplies it by its corresponding 
                # element from the kernel

                kernel_element = kernel[j+int(len(kernel)/2)][k+int(len(kernel)/2)]
                new_pixel += val * kernel_element
        # at the end of the calculation for each pixel,
        # it adds the new pixel element to final_pixels
        final_pixels[i] = new_pixel

    # returning result, new dictionary in image format
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": final_pixels,
    }

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i in range(len(image["pixels"])):
        image["pixels"][i] = round(image["pixels"][i])
        if image["pixels"][i] < 0:
            image["pixels"][i] = 0
        elif image["pixels"][i] > 255:
            image["pixels"][i] = 255



# FILTERS

def get_kernel(kernel_size):
    """
    returns a kernel based on the size.
    All elements add to 1, and it is a sizeXsize kernel
    n = 2
    .25 .25
    .25 .25

    n = 3
    .1111 .1111 .1111
    .1111 .1111 .1111
    .1111 .1111 .1111
    """

    return [[(1/kernel_size**2) for j in range(kernel_size)] for i in range(kernel_size)]


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = get_kernel(kernel_size)
    # then compute the correlation of the input image with that kernel
    corr = correlate(image, kernel, "extend")
    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(corr)
    return corr

def sharpened(image,kernel_size):
    """
    takes the image, blurs it, enumerates over it, and makes a new image according to
    the formula 2*pixel - blur(pixel)
    """
    blurred_image = blurred(image, kernel_size)
    new_image_pixels = []
    for i, pixel in enumerate(image["pixels"]): # do this for each pixel
        new_image_pixels.append(2*pixel - blurred_image["pixels"][i])

    # make new image dict to return
    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": new_image_pixels,
    }
    round_and_clip_image(new_image) # ensure it's a valid image
    return new_image

def edges(image):
    """
    find o_1 and o_2 based on the kernels given, and then calculate
    the new pixel based on formula given. then turns it into an image dict
    and makes sure its values are valid with round_and_clip_image.
    """
    kernel_1 = [[-1, -2, -1], [0,0,0], [1,2,1]]
    kernel_2 = [[-1,0,1], [-2,0,2], [-1,0,1]]

    o_1 = correlate(image, kernel_1, "extend")
    o_2 = correlate(image, kernel_2, "extend")
    final_pixels = []
    # iterates over each pixel using enumerate (more efficient than range)
    for i, pixel in enumerate(image["pixels"]):
        # uses formula from the lab
        new_pixel = math.sqrt((o_1["pixels"][i])**2+(o_2["pixels"][i])**2)
        final_pixels.append(new_pixel)

    new_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": final_pixels,
    }
    round_and_clip_image(new_image) # ensure it's a valid image
    return new_image
# HELPER FUNCTIONS FOR DISPLAYING, LOADING, AND SAVING IMAGES

def print_greyscale_values(image):
    """
    Given a greyscale image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that pixel values that are floats will be rounded to the nearest int.
    """
    out = f"Greyscale image with {image['height']} rows"
    out += f" and {image['width']} columns:\n "
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        val = str(round(pixel))
        space_vals.append((col, val))
        space_sizes[col] = max(len(val), space_sizes.get(col, 2))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, val) in space_vals:
        out += f"{val.center(space_sizes[col])} "
        if col == image["width"]-1:
            out += "\n "
    print(out)


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # cat blur
    cat = load_greyscale_image("test_images/cat.png")
    cat_blurred = blurred(cat, 13)
    save_greyscale_image(cat_blurred, "cat_blurred.png")

    # python sharp
    python = load_greyscale_image("test_images/python.png")
    python_sharp = sharpened(python, 11)
    save_greyscale_image(python_sharp, "python_sharp.png")

    # construct edges
    construct = load_greyscale_image("test_images/construct.png")
    construct_edges = edges(construct)
    save_greyscale_image(construct_edges, "construct_edges.png")

    # pigbird correlate calls:
    pigbird = load_greyscale_image("test_images/pigbird.png")
    kernel = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    corr_pigbird = correlate(pigbird,kernel, "zero")
    round_and_clip_image(corr_pigbird)
    save_greyscale_image(corr_pigbird, "pigbird_completed.png")

    corr_pigbird_extend = correlate(pigbird,kernel, "extend")
    round_and_clip_image(corr_pigbird_extend)
    save_greyscale_image(corr_pigbird_extend, "pigbird_completed2.png")

    corr_pigbird_wrap = correlate(pigbird,kernel, "wrap")
    round_and_clip_image(corr_pigbird_wrap)
    save_greyscale_image(corr_pigbird_wrap, "pigbird_completed3.png")

    # bluegill invert calls:
    bluegill = load_greyscale_image("test_images/bluegill.png")
    inverted_bluegill = inverted(bluegill)
    save_greyscale_image(inverted_bluegill, "bluegill_completed.png")
    
