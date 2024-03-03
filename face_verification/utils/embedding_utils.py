import torch
import torch.nn.functional as F


def find_closest_embedding(input_embedding, embedding_list, distance_type='cosine', verbose: bool = False):
    """
    Find the closest embedding in a list of embeddings using cosine similarity or Euclidean distance.

    Args:
      input_embedding: The input embedding to compare (tensor).
      embedding_list: A list of embeddings to compare against (list of tensors).
      distance_type: Type of distance measurement, either 'cosine' or 'euclidean'.

    Returns:
      The index of the closest embedding in the list and the corresponding similarity/distance score.
    """

    if distance_type not in ['cosine', 'euclidean']:
        raise ValueError("Invalid distance_type. Supported values are 'cosine' or 'euclidean'.")

    closest_index = None
    min_distance = float('inf') if distance_type == 'euclidean' else float('-inf')

    # Normalize input_embedding if using cosine similarity
    if distance_type == 'cosine':
        input_embedding = F.normalize(input_embedding, p=2, dim=1)[0]

    if verbose: print(input_embedding)

    for i, embedding in enumerate(embedding_list):
        # Normalize each embedding in the list if using cosine similarity
        if distance_type == 'cosine':
            embedding = F.normalize(embedding, p=2, dim=0)
        
        if verbose: print(embedding)

        # Calculate similarity or distance
        if distance_type == 'cosine':
            measure = F.cosine_similarity(input_embedding, embedding, dim=0)
        else:  # distance_type == 'euclidean'
            measure = torch.norm(input_embedding - embedding, p=2)

        if (distance_type == 'cosine' and measure > min_distance) or (distance_type == 'euclidean' and measure < min_distance):
            min_distance = measure
            closest_index = i

    return closest_index, min_distance.item()