def get_default_box_count(aspect_ratios, has_extra_box_for_ar_1=True):
    '''
    Compute the number of default boxes for each grid cell based on number of aspect ratios.
    :param aspect_ratios: List of different aspect ratios of default boxes
    :param has_extra_box_for_ar_1: Boolean for determining presence of extra box for aspect ratio 1
    :return: Integer count of number of default boxes
    '''
    assert(type(aspect_ratios) is list, 'Expected type: "list" for parameter: "aspect_ratios"')

    return len(aspect_ratios) + int((1.0 in aspect_ratios) and has_extra_box_for_ar_1)
