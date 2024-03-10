def remove_small_discontinuous_edges(edge_map, min_contour_area=100):
    # Convert edge map to binary image
    binary_edge_map = np.uint8(edge_map > 0)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the areas of contours
    contour_area_mask = np.zeros_like(edge_map)

    # Filter contours based on area
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            # Fill the contour area in the mask
            cv2.drawContours(contour_area_mask, [contour], -1, 255, -1)

    # Use the mask to keep only larger continuous edges
    filtered_edge_map = cv2.bitwise_and(edge_map, contour_area_mask)

    return filtered_edge_map