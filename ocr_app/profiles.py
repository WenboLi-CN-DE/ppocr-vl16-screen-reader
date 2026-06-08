FAST_OCR_PREDICT_KWARGS = {
    "use_layout_detection": False,
    "prompt_label": "ocr",
    "max_new_tokens": 1536,
    "max_pixels": 1280 * 28 * 28,
}
PRESERVE_LAYOUT_PREDICT_KWARGS = {
    "use_layout_detection": True,
    "merge_layout_blocks": True,
    "format_block_content": False,
    "max_new_tokens": 1536,
    "max_pixels": 1280 * 28 * 28,
}
MAX_FAST_IMAGE_EDGE = 1536
LOW_MEMORY_FAST_OCR_PREDICT_KWARGS = {
    **FAST_OCR_PREDICT_KWARGS,
    "max_new_tokens": 768,
    "max_pixels": 768 * 28 * 28,
}
LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS = {
    **PRESERVE_LAYOUT_PREDICT_KWARGS,
    "max_new_tokens": 768,
    "max_pixels": 768 * 28 * 28,
}
EMERGENCY_FAST_OCR_PREDICT_KWARGS = {
    **FAST_OCR_PREDICT_KWARGS,
    "max_new_tokens": 512,
    "max_pixels": 512 * 28 * 28,
}
EMERGENCY_PRESERVE_LAYOUT_PREDICT_KWARGS = {
    **PRESERVE_LAYOUT_PREDICT_KWARGS,
    "max_new_tokens": 512,
    "max_pixels": 512 * 28 * 28,
}
MAX_LOW_MEMORY_IMAGE_EDGE = 1024
MAX_EMERGENCY_IMAGE_EDGE = 768


def resize_for_ocr(image, max_edge, resample=None):
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return image
    scale = max_edge / longest_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, resample)


def select_predict_kwargs(*, preserve_layout, fast_mode, low_memory_mode):
    if preserve_layout:
        if low_memory_mode:
            return LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS.copy()
        return PRESERVE_LAYOUT_PREDICT_KWARGS.copy()
    if fast_mode:
        if low_memory_mode:
            return LOW_MEMORY_FAST_OCR_PREDICT_KWARGS.copy()
        return FAST_OCR_PREDICT_KWARGS.copy()
    return {}


def select_max_edge(*, fast_mode, preserve_layout, low_memory_mode):
    if not (fast_mode or preserve_layout):
        return None
    if low_memory_mode:
        return MAX_LOW_MEMORY_IMAGE_EDGE
    return MAX_FAST_IMAGE_EDGE


def emergency_predict_kwargs(predict_kwargs):
    if predict_kwargs.get("use_layout_detection"):
        return EMERGENCY_PRESERVE_LAYOUT_PREDICT_KWARGS.copy()
    if predict_kwargs.get("use_layout_detection") is False:
        return EMERGENCY_FAST_OCR_PREDICT_KWARGS.copy()
    return EMERGENCY_FAST_OCR_PREDICT_KWARGS.copy()
