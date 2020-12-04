CR-Vision Image Processing Cheat Sheet
==============================================

.. highlight:: python

This section is a quick overview of sample code
for essential image processing operations 
using CR-Vision library. For more detailed code 
examples, please explore the examples directory
in the source code.

The examples below freely use other
image processing libraries like ``opencv``,
``imageio``, ``scikit-image`` wherever
applicable along with the features provided
by ``cr-vision``.


Essential Imports::

    import imageio
    import cv2
    from cr import vision

Image Read/Write
--------------------

Reading image in BGR format::

    image = cv2.imread(image_path)


Reading image in RGB format::

    image = imageio.imread(image_path)


Color Space Conversion
-------------------------------

From BGR to Gray Scale::

    gray_image = vision.bgr_to_gray(image)


Resizing
------------------

Resizing operations preserve aspect ratio by default.

Resize to a specific height::

    image = vision.resize_by_height(image, target_height)


Resize to a specific width::

    image = vision.resize_by_width(image, target_width)


Resize an image if its width exceeds a given maximum width::

    image = vision.resize_by_max_width(image, max_width)


Resize to a specific width and height (aspect ratio is not preserved)::

    image = vision.resize(image, target_width, target_height)




Thresholding
---------------------

Adaptive Gaussian Thresholding on an image::

    thresholded_image = vision.adaptive_threshold_gaussian(
        gray_image, block_size=115, constant=1)



Rotations
-------------------

Rotate by a specific angle::

    from cr.vision import geom
    image = geom.rotate(image, angle)


Translations
----------------------

Translate an image by (x,y) pixels::

    from cr.vision import geom
    image = geom.translate(image, (x, y))


Logos
-----------

Adding a logo in a corner::

    from cr.vision.edits import logo
    image = cv2.imread(image_path)
    logo = cv2.imread(logo_path)
    logo = cv2.resize(logo, (0, 0), fx=0.5, fy=0.5)
    image = logo.add_logo(image, logo)


Borders
------------------


Add multiple borders of specific widths and colors::

    from cr.vision import edits
    image = edits.add_multiple_borders(image, widths=20, colors=[
        colors.BLUE, colors.RED, colors.GREEN])


Add a letterbox around an image::

    from cr.vision import edits
    image = edits.add_letter_box_pattern(image, letterbox_height)
