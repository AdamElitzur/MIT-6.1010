#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing 2
"""

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
import os
# import typing  # optional import
from PIL import Image


# COPY THE FUNCTIONS THAT YOU IMPLEMENTED IN IMAGE PROCESSING PART 1 BELOW!
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




# VARIOUS FILTERS
def split_color_image_to_greyscales(image):
    """
    takes image and loops over every pixel and splits it into red, green blue
    then returns it in 3 different lists with RGB split.
    """
    red = []
    green = []
    blue = []
    for pixel in image["pixels"]:
        red.append(pixel[0])
        green.append(pixel[1])
        blue.append(pixel[2])

    red_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": red
    }
    green_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": green
    }
    blue_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": blue
    }
    return red_image, green_image, blue_image

def combine_greyscales_to_color(red, green, blue):
    """
    takes image and loops over every pixel and combines it into a list of sets
    with R,G,B values, then combines it to a new_image and returns
    """
    new_pixels = []
    for i, pixel in enumerate(red["pixels"]):
        new_pixels.append((pixel, green["pixels"][i], blue["pixels"][i]))

    new_image = {
        "height": red["height"],
        "width": red["width"],
        "pixels": new_pixels
    }
    return new_image

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def new(image):
        # splits into greyscale images per color
        red, green, blue = split_color_image_to_greyscales(image)
        # filter all of the images
        filtered_red = filt(red)
        filtered_green = filt(green)
        filtered_blue = filt(blue)
        # combines and returns them
        return combine_greyscales_to_color(filtered_red, filtered_green, filtered_blue)
    return new


def make_blur_filter(kernel_size):
    """
    Makes a new function that takes in an image
    returns the blurred image with kernel_size from 
    make_blur_filter call
    """
    def func(image):
        return blurred(image, kernel_size)
    return func


def make_sharpen_filter(kernel_size):
    """
    Makes a new function that takes in an image
    returns the sharpened image with kernel_size from 
    make_sharpen_filter call
    """
    def func(image):
        return sharpened(image, kernel_size)
    return func


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def func(image):
        # makes a copy of the image so doesn't mutate it
        new_image = image.copy()
        # loops through filters and applies them, setting new_image
        # to result each time
        for i in filters:
            new_image = i(new_image)
        return new_image
    return func


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    raise NotImplementedError


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    raise NotImplementedError


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    raise NotImplementedError


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function) greyscale image, computes a "cumulative energy map" as described
    in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    raise NotImplementedError


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map dictionary, returns a list of the indices into
    the 'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    raise NotImplementedError


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    raise NotImplementedError


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


def print_color_values(image):
    """
    Given a color image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that RGB values will be rounded to the nearest int.
    """
    out = f"Color image with {image['height']} rows"
    out += f" and {image['width']} columns:\n"
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        for color in range(3):
            val = str(round(pixel[color]))
            space_vals.append((col, color, val))
            space_sizes[(col, color)] = max(len(val), space_sizes.get((col, color), 0))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, color, val) in space_vals:
        space_val = val.center(space_sizes[(col, color)])
        if color == 0:
            out += f" ({space_val}"
        elif color == 1:
            out += f" {space_val} "
        else:
            out += f"{space_val})"
        if col == image["width"]-1 and color == 2:
            out += "\n"
    print(out)


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
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
    by the 'mode' parameter.
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

    # # color_inverted cat:
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # color_cat = color_inverted(load_color_image("test_images/cat.png"))
    # save_color_image(color_cat, "inverted_cat.png")

    # # blur python and sharp sparrow
    # color_blur = color_filter_from_greyscale_filter(make_blur_filter(9))
    # blur_python = color_blur(load_color_image("test_images/python.png"))
    # save_color_image(blur_python, "blurred_python.png")

    # color_sharpened = color_filter_from_greyscale_filter(make_sharpen_filter(7))
    # sharpened_sparrow = color_sharpened(load_color_image("test_images/sparrowchick.png"))
    # save_color_image(sharpened_sparrow, "sharpened_sparrow.png")

    # filter_cascase frog
    filter1 = color_filter_from_greyscale_filter(edges)
    filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    filt = filter_cascade([filter1, filter1, filter2, filter1])

    filtered_frog = filt(load_color_image("test_images/frog.png"))
    save_color_image(filtered_frog, "filtered_frog.png")