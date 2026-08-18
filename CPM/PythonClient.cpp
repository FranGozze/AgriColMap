#include "PythonClient.hpp"

// base64 same as before
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const std::vector<uchar>& data) {
    std::string result;
    int val = 0, valb = -6;
    for (uchar c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) result.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (result.size() % 4) result.push_back('=');
    return result;
}

PythonClient::PythonClient(const std::string& addr)
    : address(addr) {}


std::string PythonClient::encodeFImage(const FImage& img) {
    
    return encodeImage(ImageIO::CvmatFromPixels(img.pData,img.width(),img.height(),img.nchannels()));
}

std::string PythonClient::encodeImage(const cv::Mat& img) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", img, buffer);
    return base64_encode(buffer);
}

void PythonClient::extract(
    const FImage& imgExg,
    const FImage& imgElev,
    float cloud_ratio,
    const int method,
    UCImage& outFtImg_Exg,
    UCImage& outFtImg_Elev
) {
    int w = imgExg.width();
    int h = imgExg.height();

    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect(address);

    nlohmann::json request;
    request["img_exg"] = encodeFImage(imgExg);
    request["img_elev"] = encodeFImage(imgElev);
    request["cloud_ratio"] = cloud_ratio;
    request["id_method"] = method;

    std::string req_str = request.dump();
    socket.send(zmq::buffer(req_str), zmq::send_flags::none);


    zmq::message_t header_msg;
    zmq::message_t exg_msg;
    zmq::message_t elev_msg;

    socket.recv(header_msg);
    socket.recv(exg_msg);
    socket.recv(elev_msg);

    // ---- parse header ----
    int dims[6];
    std::memcpy(dims, header_msg.data(), sizeof(dims));

    int H = dims[0];
    int W = dims[1];
    int C1 = dims[2];
    int H2 = dims[3];
    int W2 = dims[4];
    int C2 = dims[5];
    // std::cout << "Received header: H=" << H << ", W=" << W << ", C1=" << C1
    //           << ", H2=" << H2 << ", W2=" << W2 << ", C2=" << C2 << std::endl;
    // ---- allocate outputs ----
    outFtImg_Exg.allocate(W, H, C1);
    outFtImg_Elev.allocate(W2, H2, C2);
    // std::cout << "Allocated output image memory" << std::endl;

    // ---- copy directly into UCImage ----
    size_t size_exg = H * W * C1 * sizeof(float);
    size_t size_elev = H2 * W2 * C2 * sizeof(float);

    // const float* elev_ptr = static_cast<float*>(elev_msg.data());
    // std::cout << "Received feature data: exg size=" << size_exg << ", elev size=" << size_elev << std::endl;

    std::memcpy(outFtImg_Exg.pData, exg_msg.data(), H*W*C1);
    // std::cout << "Copied EXG feature data" << std::endl;

    std::memcpy(outFtImg_Elev.pData, elev_msg.data(), H2*W2*C2);
    
    // for (int i = 0; i < H2 * W2 * C2; i++) {
    //     outFtImg_Elev.pData[i] =
    //         static_cast<unsigned char>(std::round(elev_ptr[i] * 255));
    // }

    // std::cout << "Copied ELEV feature data" << std::endl;

}



// MatchResult PythonClient::match(const FImage& img1, const FImage& img2, const int mode) {
//     zmq::context_t context(1);
//     zmq::socket_t socket(context, ZMQ_REQ);

//     socket.connect(address);

//     nlohmann::json request;
//     request["img1"] = encodeImage(img1);
//     request["img2"] = encodeImage(img2);
//     request["mode"] = mode;

//     std::string req_str = request.dump();
//     socket.send(zmq::buffer(req_str), zmq::send_flags::none);

//     zmq::message_t reply;
//     socket.recv(reply);

//     auto response = nlohmann::json::parse(
//         std::string(static_cast<char*>(reply.data()), reply.size())
//     );

//     MatchResult result;

//     for (auto& p : response["pts1"]) {
//         result.pts1.emplace_back(p[0], p[1]);
//     }
//     for (auto& p : response["pts2"]) {
//         result.pts2.emplace_back(p[0], p[1]);
//     }

//     return result;
// }