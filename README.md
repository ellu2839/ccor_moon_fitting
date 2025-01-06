In summary, we found that with Chris' fudged WCS, we were successfully able to use SM_ANGLE to determine whether or not the moon should be in the 11deg (diameter, 5.5deg radius) FOV. 
Using Chris' code, we are also able to determine moon location and radius, as well as distance from the center of the sun.
We are also technically able to visually edge-detect the moon using bidirectional gaussian blurring + Canny edge detection. This is an imperfect solution due to the low contrast of the
moon relative to the rest of the image. For the most part, this is fairly accurate, although not more accurate than Chris' code. It does get thrown off in super low contrast images, or where
the algorithm picks up craters and other texture on the moon and changes where it thinks the centroid is.

This project will be useful later when we attempt to use this for photometry applications, for dealing with scattered light from the moon/earth/sun/satellite.
