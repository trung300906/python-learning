#include <bits/stdc++.h>
using namespace std;

int main() {
    srand(time(NULL));
    ofstream data_file("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/logistic_regression/data.txt");
    ofstream theta_file("/run/media/trung/hdddrive/CODE/python-learning/AI-LEARNING/logistic_regression/theta.txt");

    if (!data_file || !theta_file) {
        cerr << "Không thể mở file!" << endl;
        return 1;
    }

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(50.0, 15.0);  // Phân phối chuẩn, mean = 50, stddev = 15
    bernoulli_distribution noise(0.01);  // 1% khả năng lật nhãn

    long long num_samples = 1920*1080;
    int num_features = 1;  // Giả sử chỉ có 1 feature

    // Tạo và ghi dữ liệu
    for (long long i = 0; i < num_samples; i++) {
        double x = dist(gen);
        int y = (x > 50) ? 1 : 0;

        // Thêm nhiễu vào nhãn
        if (noise(gen)) y = 1 - y;

        data_file << x << " " << y << endl;
    }

    data_file.close();

    // Sinh ngẫu nhiên theta (dài vừa đủ)
    uniform_real_distribution<double> theta_dist(-1.0, 1.0); // Giá trị từ -1 đến 1
    vector<double> theta(num_features + 1);  // Thêm 1 cho bias

    for (double &t : theta) {
        t = theta_dist(gen);
        theta_file << t << " ";
    }

    theta_file.close();

    cout << "Dữ liệu và theta đã được ghi vào file thành công." << endl;
    
    return 0;
}
