#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
# include<ctime>
# include<iostream>
# include<random> 
# include<fstream>




using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 3) {
        cout << "Usage: ./rgb_to_rgba <input_image_path> <output_image_path>" << endl;
        return -1;
    }
    // 从命令行参数中获取输入和输出图像路径
    string input_image_path = argv[1];
    string output_image_path = argv[2];

    // 读取RGB格式图像
    Mat image_rgb = imread(input_image_path);

    if (image_rgb.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // 创建一个包含透明度通道的Mat对象
    Mat image_rgba(image_rgb.rows, image_rgb.cols, CV_8UC4);

    // 将RGB图像复制到RGBA图像中
    cvtColor(image_rgb, image_rgba, COLOR_RGB2RGBA);

    // 可以手动设置透明度通道的值（在这个例子中设置为255，即完全不透明）
    for (int y = 0; y < image_rgba.rows; ++y) {
        for (int x = 0; x < image_rgba.cols; ++x) {
            Vec4b& pixel = image_rgba.at<Vec4b>(y, x);
            pixel[3] = 255;  // 设置透明度通道值
        }
    }
    // 保存转换后的图像
    imwrite(output_image_path, image_rgba);

    return 0;
}
