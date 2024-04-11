#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

// 互斥量用于保护共享资源
std::mutex mtx;

// 读取文件内容并存储到数据结构中
void read_file(const std::string& filename, std::vector<std::string>& file_contents) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        // 临界区，需要保护共享资源
        std::lock_guard<std::mutex> lock(mtx);
        file_contents.push_back(buffer.str());
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

// 合并多个文件内容
std::string merge_files(const std::vector<std::string>& file_contents) {
    std::stringstream merged_content;
    for (const auto& content : file_contents) {
        merged_content << content;
    }
    return merged_content.str();
}

// 将合并后的内容写入CSV文件
void write_to_csv(const std::string& merged_content, const std::string& output_filename) {
    std::ofstream output_file(output_filename);
    if (output_file.is_open()) {
        output_file << merged_content;
        output_file.close();
    } else {
        std::cerr << "Failed to open output file: " << output_filename << std::endl;
    }
}

int main() {
    std::vector<std::string> file_contents;
    std::vector<std::thread> threads;

    // 文件列表
    std::vector<std::string> filenames = {"/home/wxcwxc/ctensorRt/C++Project/Cquant/file1.txt",
     "/home/wxcwxc/ctensorRt/C++Project/Cquant/file2.txt", "/home/wxcwxc/ctensorRt/C++Project/Cquant/file3.txt"};

    // 创建并启动多个线程读取文件
    for (const auto& filename : filenames) {
        threads.emplace_back(read_file, filename, std::ref(file_contents));
    }
    // 等待所有线程结束
    for (auto& thread : threads) {
        thread.join();
    }
    // 合并文件内容
    std::string merged_content = merge_files(file_contents);

    // 将合并后的内容写入CSV文件
    write_to_csv(merged_content, "output.csv");

    return 0;
}
