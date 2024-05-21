#include <iostream>
#include <thread>

#include <pcl/common/angles.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iomanip> // for setw, setfill

pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main() {
    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>),
                                        cloud1(new pcl::PointCloud<pcl::PointXYZ>),
                                        cloud2(new pcl::PointCloud<pcl::PointXYZ>),
                                        cloud3(new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("cloud_cluster_0000.pcd", *cloud0);
    reader.read("cloud_cluster_0001.pcd", *cloud1);
    reader.read("cloud_cluster_0002.pcd", *cloud2);
    reader.read("cloud_cluster_0003.pcd", *cloud3);

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    *point_cloud_ptr += *cloud0;
    *point_cloud_ptr += *cloud1;
    *point_cloud_ptr += *cloud2;
    *point_cloud_ptr += *cloud3;

    pcl::visualization::PCLVisualizer::Ptr viewer = simpleVis(point_cloud_ptr);

    point_cloud_ptr->width = point_cloud_ptr->size();
    point_cloud_ptr->height = 1;
    point_cloud_ptr->is_dense = true;

    while(true) {
        viewer->updatePointCloud(point_cloud_ptr, "sample cloud");
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

  return 0;
}
