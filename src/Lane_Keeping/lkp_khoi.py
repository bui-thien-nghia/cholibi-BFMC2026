import numpy as np

def find_x_at_y(poly, target_y, img_width=640):
    if poly is None:
        return None
    coeffs = poly.coeffs.copy()
    coeffs[-1] -= target_y
    roots = np.poly1d(coeffs).roots
    valid_x = []
    for val in roots:
        if not np.iscomplex(val):
            x_val = val.real
            if 0 <= x_val <= img_width:
                valid_x.append(x_val)
    if len(valid_x) > 0:
        return valid_x[0]
    return None



