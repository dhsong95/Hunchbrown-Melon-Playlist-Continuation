def normalize_zero_to_one(rating):
    rating_min = rating.min(-1)
    rating_max = rating.max(-1)
    if rating_max != 0:
        rating = (rating - rating_min) / (rating_max - rating_min)
    return rating
