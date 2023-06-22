import os
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import glob
import torch


class ImageDepth:
    def __init__(self, image):
        self.image = image
        self.feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-nyu')

        self.resize_image()
        self.getDepthImage()

    def resize_image(self):
        new_height = 480 if self.image.height > 480 else self.image.height
        new_height -= (new_height % 32)
        new_width = int(new_height * self.image.width / self.image.height)
        diff = new_width % 32
        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size = (new_width, new_height)
        self.image = self.image.resize(new_size)

    def getDepthImage(self):
        inputs = self.feature_extractor(images=self.image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        pad = 16
        self.output = predicted_depth.squeeze().cpu().numpy() * 1000.0
        self.output = self.output[pad:-pad, pad:-pad]
        self.image = self.image.crop((pad, pad, self.image.width - pad, self.image.height - pad))

    def get3d(self):
        width, height = self.image.size
        depth_image = (self.output * 255 / np.max(self.output)).astype(np.uint8)
        image_o3d = o3d.geometry.Image(np.array(self.image))
        depth_o3d = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                        convert_rgb_to_intensity=False)

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

        return point_cloud


def main():
    arr_img = []
    for name in glob.glob('./PhotoAutour/*.jpg'):
        n = Image.open(name)
        arr_img.append(n)

    point_clouds = []
    for img in arr_img[::50]:
        depth = ImageDepth(img)
        point_clouds.append(depth.get3d())

    combined_point_cloud = o3d.geometry.PointCloud()
    for pc in point_clouds:
        pc.translate((-np.mean(pc.points, axis=0)).tolist())  # Translate to center of mass
        
        # Scale the points to unit size
        max_distance = np.max(np.linalg.norm(np.asarray(pc.points), axis=1))
        pc.points = o3d.utility.Vector3dVector(np.asarray(pc.points) / max_distance)

        combined_point_cloud += pc

    # Génération du maillage avec la méthode de Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_point_cloud, depth=9)

    # Simplification du maillage
    mesh = mesh.simplify_quadric_decimation(100000)

    # Sauvegarde du maillage en fichier OBJ
    o3d.io.write_triangle_mesh("result.obj", mesh)

    o3d.visualization.draw_geometries([combined_point_cloud])


if __name__ == '__main__':
    main()