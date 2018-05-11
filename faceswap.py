#! /usr/bin/env python3

import argparse
import os
import time

import numpy as np
import cv2
import dlib

here = os.path.abspath(os.path.dirname(__file__))
_predictor_path = 'shape_predictor_68_face_landmarks.dat'
_casc_path = 'haarcascade_frontalface_alt.xml'
predictor_path = os.path.join(here, _predictor_path)
casc_path = os.path.join(here, _casc_path)


def applyAffineTransform(src, srcTri, dstTri, size):
    '''
    Apply affine transform calculated using srcTri and dstTri to src and
    output an image of size.
    '''
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    dst = cv2.warpAffine(
        src,
        warpMat, (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return dst


def rectContains(rect, point):
    x, y, w, h = rect
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert(tuple(p))

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    count = 0

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(
                rect, pt2) and rectContains(rect, pt3):
            count = count + 1
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(
                            pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []
    return delaunayTri


def warpTriangle(img1, img2, t1, t2):
    '''
    Warps and alpha blends triangular regions from img1 and img2 to img
    '''
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[
        r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]
         ] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


class FPSReporter:
    def __init__(self):
        self.last = None

    def __call__(self):
        now = time.perf_counter()
        if self.last is not None:
            print(
                'time={:.2g}, fps={:.2g}'.
                format(now - self.last, 1 / (now - self.last))
            )
        self.last = now


fps = FPSReporter()


def init():
    global casade_classifier, detector, predictor
    casade_classifier = cv2.CascadeClassifier(casc_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)


def detect_faces_fast(img):
    boxes = casade_classifier.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    return [
        dlib.rectangle(*map(int, [x, y, x + w, y + h]))
        for (x, y, w, h) in boxes
    ]


def detect_faces_slow(img):
    return detector(img, 1)


def predict_landmark(img, rect):
    return [(point.x, point.y) for point in predictor(img, rect).parts()]


def check_bound(img, points):
    '''
    returns true if all points are in the image
    '''
    height, width, *_ = img.shape
    return all(0 <= x < width and 0 <= y < height for (x, y) in points)


def get_convex_hull_indexes(points):
    return cv2.convexHull(np.array(points), returnPoints=False)[:, 0]


face68_reflections = np.array(
    [
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24,
        23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44,
        43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59,
        58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65
    ]
)
left_face_indexes = np.array(
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 66, 62, 51, 33, 30, 29, 28, 27, 21, 20,
        19, 18, 17
    ]
)
right_face_indexes = np.array(
    [
        8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 27, 28, 29, 30,
        33, 51, 62, 66, 57
    ]
)


USE_CONVEX_HULL = False
AUTO_REFLECT = True


def put_face(
        srcimg,
        srcpoints,
        dstimg,
        dstpoints,
):
    srcpoints = np.array(srcpoints)
    dstpoints = np.array(dstpoints)
    dsth, dstw, *_ = dstimg.shape
    dstbounds = (0, 0, dstw, dsth)
    # area test & reflect
    if AUTO_REFLECT:
        srcl = cv2.contourArea(srcpoints[left_face_indexes])
        srcr = cv2.contourArea(srcpoints[right_face_indexes])
        dstl = cv2.contourArea(dstpoints[left_face_indexes])
        dstr = cv2.contourArea(dstpoints[right_face_indexes])
        if (srcl < srcr) == (dstl > dstr):
            srcpoints = srcpoints[face68_reflections]
    # find convex hull
    hull_indexes = get_convex_hull_indexes(dstpoints)
    dsthull = dstpoints[hull_indexes]
    if USE_CONVEX_HULL:
        srcpoints = srcpoints[hull_indexes]
        dstpoints = dsthull
    # delaunay triangulation
    try:
        delaunay_triangles = calculateDelaunayTriangles(dstbounds, dstpoints)
    except cv2.error:
        return dstimg
    assert delaunay_triangles
    warpedimg = np.copy(dstimg)
    # affine transformation to Delaunay triangles
    for triangle in delaunay_triangles:
        warpTriangle(
            srcimg,
            warpedimg,
            [srcpoints[point] for point in triangle],
            [dstpoints[point] for point in triangle],
        )
    # mask
    hull8U = [(p[0], p[1]) for p in dsthull]
    mask = np.zeros(dstimg.shape, dtype=dstimg.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(np.float32([dsthull]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    return cv2.seamlessClone(
        np.uint8(warpedimg), dstimg, mask, center, cv2.NORMAL_CLONE
    )


def img2video(srcfile, capture=0):
    srcimg = cv2.imread(srcfile)
    srcfaces = detect_faces_slow(srcimg)
    srcpoints = predict_landmark(srcimg, srcfaces[0])
    cap = cv2.VideoCapture(capture)
    while True:
        fps()
        ret, dstimg = cap.read()
        cv2.imshow('original', dstimg)
        assert ret
        dstfaces = detect_faces_fast(dstimg)
        for rect in dstfaces:
            dstpoints = predict_landmark(dstimg, rect)
            if not check_bound(dstimg, dstpoints):
                continue
            dstimg = put_face(srcimg, srcpoints, dstimg, dstpoints)
        cv2.imshow('face swapped', dstimg)
        cv2.waitKey(1)


def imgswap(filename):
    img = cv2.imread(filename)
    faces = detect_faces_slow(img)
    pointss = [predict_landmark(img, rect) for rect in faces]
    pointss = [points for points in pointss if check_bound(img, points)]
    srcimg = img.copy()
    for pa, pb in zip(pointss, pointss[1:] + [pointss[0]]):
        img = put_face(srcimg, pa, img, pb)
    return img


def img2img(srcfile, dstfile):
    srcimg = cv2.imread(srcfile)
    srcfaces = detect_faces_slow(srcimg)
    srcpoints = predict_landmark(srcimg, srcfaces[0])
    dstimg = cv2.imread(dstfile)
    dstfaces = detect_faces_slow(dstimg)
    for rect in dstfaces:
        dstpoints = predict_landmark(dstimg, rect)
        if not check_bound(dstimg, dstpoints):
            continue
        dstimg = put_face(srcimg, srcpoints, dstimg, dstpoints)
    return dstimg


def get_parser():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
    )
    parser.add_argument(
        'source',
        help='the source image file',
    )
    group = (
        parser.add_argument_group('destination options')
        .add_mutually_exclusive_group(required=True)
    )
    group.add_argument(
        '--self',
        action='store_true',
        help='swap the faces in the source image',
    )
    group.add_argument(
        '--dst',
        metavar='filename',
        help='swap the face from the source image into this image file',
    )
    group.add_argument(
        '--camera',
        type=int,
        dest='video',
        metavar='index',
        help='swap the face from the source image into this video device',
    )
    group.add_argument(
        '--video',
        metavar='filename',
        help='swap the face from the source image into this video file',
    )
    parser.add_argument(
        '--save',
        metavar='filename',
        help='save the swapped image to this file',
    )
    parser.add_argument(
        '--noshow', action='store_true', help="don't cv2.imshow"
    )
    parser.add_argument(
        '--directpaste',
        action='store_true',
        help='paste convex hull instead of feature points',
    )
    parser.add_argument(
        '--noreflect',
        action='store_true',
        help="do not reflect feature points vertically even if two faces are "
        "facing different sides"
    )
    return parser


if __name__ == '__main__':
    options = get_parser().parse_args()
    init()
    USE_CONVEX_HULL = options.directpaste
    AUTO_REFLECT = not options.noreflect
    if options.self or options.dst is not None:
        if options.self:
            result = imgswap(options.source)
        else:
            result = img2img(options.source, options.dst)
        if options.save is not None:
            cv2.imwrite(options.save, result)
        if not options.noshow:
            cv2.imshow('swap', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif options.camera is not None or options.video is not None:
        img2video(options.video)
    else:
        assert False, "Shouldn't be here :("
