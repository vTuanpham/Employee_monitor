from PIL import Image


def pad_largest_images(pil_images):
  """
  Pads the largest images in a list of PIL images to have the same size.

  Args:
    pil_images: A list of PIL images.

  Returns:
    A list of padded PIL images.
  """

  # Find the largest image size.
  max_width = max(image.width for image in pil_images)
  max_height = max(image.height for image in pil_images)

  # Pad each image to the largest size.
  padded_images = []
  for image in pil_images:
    padded_image = Image.new('RGB', (max_width, max_height), (0, 0, 0))
    padded_image.paste(image, (0, 0))
    padded_images.append(padded_image)

  return padded_images