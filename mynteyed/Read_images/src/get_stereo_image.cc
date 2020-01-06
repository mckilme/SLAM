#include <iostream>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>

#include "mynteyed/camera.h"
#include "mynteyed/utils.h"

#include "util/cam_utils.h"
#include "util/counter.h"
#include "util/cv_painter.h"

using namespace std;
using namespace Eigen;

MYNTEYE_USE_NAMESPACE

typedef Matrix <double,6,1> Vector6d;

//void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 640, 480);//窗口大小
    glEnable(GL_DEPTH_TEST);//打开深度测试模式，需要绘制透明图片时需要关闭
    glEnable(GL_BLEND);//打开混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 468, 468, 320, 240, 0.3, 100),//定义相机投影模型
        pangolin::ModelViewLookAt(0, 0, -1.0, 0, 0, 0, 0.0, -1.0, 0.0)//定义观测方位向量
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -640.0f / 480.0f)//与oengl的viewpoint有关
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            //glColor3f(p[3], p[3], p[3]);
            glColor3f(p[3], p[4], p[5]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

int main(int argc, char const* argv[]) {
  Camera cam;
  
  double lfx = 519.289184,lfy = 519.027466,lcx = 316.846619,lcy = 247.905777;
  //double rfx = 520.203979,rfy = 520.055542,rcx = 638.842651,rcy = 368.298767;
  double b = 0.12;
  cv::Mat left;
  cv::Mat right;
  cv::Mat disparity_sgbm, disparity;
  
  DeviceInfo dev_info;
  if (!util::select(cam, &dev_info)) {
    return 1;
  }
  util::print_stream_infos(cam, dev_info.index);

  std::cout << "Open device: " << dev_info.index << ", "
      << dev_info.name << std::endl << std::endl;

  OpenParams params(dev_info.index);
  {
    // Framerate: 30(default), [0,60], [30](STREAM_2560x720)
    params.framerate = 30;

    // Device mode, default DEVICE_ALL
    //   DEVICE_COLOR: IMAGE_LEFT_COLOR ✓ IMAGE_RIGHT_COLOR ? IMAGE_DEPTH x
    //   DEVICE_DEPTH: IMAGE_LEFT_COLOR x IMAGE_RIGHT_COLOR x IMAGE_DEPTH ✓
    //   DEVICE_ALL:   IMAGE_LEFT_COLOR ✓ IMAGE_RIGHT_COLOR ? IMAGE_DEPTH ✓
    // Note: ✓: available, x: unavailable, ?: depends on #stream_mode
    // params.dev_mode = DeviceMode::DEVICE_ALL;

    // Color mode: raw(default), rectified
    // params.color_mode = ColorMode::COLOR_RECTIFIED;

    // Depth mode: colorful(default), gray, raw
    // params.depth_mode = DepthMode::DEPTH_GRAY;

    // Stream mode: left color only
    // params.stream_mode = StreamMode::STREAM_640x480;  // vga
    // params.stream_mode = StreamMode::STREAM_1280x720;  // hd
    // Stream mode: left+right color
    params.stream_mode = StreamMode::STREAM_1280x480;  // vga
    // params.stream_mode = StreamMode::STREAM_2560x720;  // hd

    // Auto-exposure: true(default), false
    // params.state_ae = false;

    // Auto-white balance: true(default), false
    // params.state_awb = false;

    // IR Depth Only: true, false(default)
    // Note: IR Depth Only mode support frame rate between 15fps and 30fps.
    //     When dev_mode != DeviceMode::DEVICE_ALL,
    //       IR Depth Only mode not be supported.
    //     When stream_mode == StreamMode::STREAM_2560x720,
    //       frame rate only be 15fps in this mode.
    //     When frame rate less than 15fps or greater than 30fps,
    //       IR Depth Only mode will be not available.
    // params.ir_depth_only = true;

    // Infrared intensity: 0(default), [0,10]
    params.ir_intensity = 0;

    // Colour depth image, default 5000. [0, 16384]
    params.colour_depth_value = 5000;
  }

  // Enable what process logics
  // cam.EnableProcessMode(ProcessMode::PROC_IMU_ALL);

  // Enable image infos
  cam.EnableImageInfo(true);

  cam.Open(params);

  std::cout << std::endl;
  if (!cam.IsOpened()) {
    std::cerr << "Error: Open camera failed" << std::endl;
    return 1;
  }
  std::cout << "Open device success" << std::endl << std::endl;

  std::cout << "Press ESC/Q on Windows to terminate" << std::endl;

  bool is_left_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_LEFT_COLOR);
  bool is_right_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_RIGHT_COLOR);
  bool is_depth_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_DEPTH);

  if (is_left_ok) cv::namedWindow("left color");
  if (is_right_ok) cv::namedWindow("right color");
  if (is_depth_ok) cv::namedWindow("disparity");

  CVPainter painter;
  util::Counter counter;
  for (;;) {
    cam.WaitForStream();
    counter.Update();
    
    if (is_left_ok) {
      auto left_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
      if (left_color.img) {
        //cv::Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
        left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
        painter.DrawSize(left, CVPainter::TOP_LEFT);
        painter.DrawStreamData(left, left_color, CVPainter::TOP_RIGHT);
        painter.DrawInformation(left, util::to_string(counter.fps()),
            CVPainter::BOTTOM_RIGHT);
        cv::imshow("left color", left);
      }
    }
    
    if (is_right_ok) {
      auto right_color = cam.GetStreamData(ImageType::IMAGE_RIGHT_COLOR);
      if (right_color.img) {
        //cv::Mat right = right_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
        right = right_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
        painter.DrawSize(right, CVPainter::TOP_LEFT);
        painter.DrawStreamData(right, right_color, CVPainter::TOP_RIGHT);
        cv::imshow("right color", right);
      }
    }

    if (is_depth_ok) {
        /*
      auto image_depth = cam.GetStreamData(ImageType::IMAGE_DEPTH);
      if (image_depth.img) {
        cv::Mat depth;
        if (params.depth_mode == DepthMode::DEPTH_COLORFUL) {
          depth = image_depth.img->To(ImageFormat::DEPTH_BGR)->ToMat();
        } else {
          depth = image_depth.img->ToMat();
        }
        painter.DrawSize(depth, CVPainter::TOP_LEFT);
        painter.DrawStreamData(depth, image_depth, CVPainter::TOP_RIGHT);
        cv::imshow("depth", depth);
      }*/
      //cv::Mat leftGray;
      //cv::Mat rightGray;
      //cv::cvtColor(left, leftGray, CV_BGR2GRAY);
      //cv::cvtColor(right, rightGray, CV_BGR2GRAY);
      cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,96,9,8*9*9,32*9*9,1,63,10,100,32);//神奇的参数
      sgbm->compute(left,right,disparity_sgbm);
      //sgbm->compute(leftGray,rightGray,disparity_sgbm);
      disparity_sgbm.convertTo(disparity, CV_32F,1.0 / 16.0f);
      cv::imshow("disparity",disparity / 96.0);
    }

    

    
    char key = static_cast<char>(cv::waitKey(1));
    if( key == 'q' || key == 'Q'){
        //vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
        for(int v = 0 ; v < left.rows ;v++){
            for(int u = 0 ; u < left.cols ;u++){
                if(disparity.at<float>(v, u) <= 10 || disparity.at<float>(v, u) >= 96.0)
                    continue;
                //Vector4d point(0,0,0,left.at<uchar>(v, u) / 255.0);
                Vector6d point; 
                point<< 0,0,0,left.at<cv::Vec3b>(v, u)[2] / 255.0,left.at<cv::Vec3b>(v, u)[1] / 255.0,left.at<cv::Vec3b>(v, u)[0] / 255.0;
                double x = (u - lcx)/lfx;
                double y = (v - lcy)/lfy;
                double depth = lfx * b / (disparity.at<float>(v, u));
                point[0] = x*depth;
                point[1] = y*depth;
                point[2] = depth;
                
                pointcloud.push_back(point);
            }
        }
        showPointCloud(pointcloud);
        continue;
    }
    else if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
      break;
    }
  }

  cam.Close();
  cv::destroyAllWindows();
  return 0;
}
