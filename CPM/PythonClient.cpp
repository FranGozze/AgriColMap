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
    int w = img.width();
    int h = img.height();
    int c = img.nchannels();

    std::vector<unsigned char> buffer;
    buffer.reserve(w * h * c);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int idx = (i * w + j) * c;

            for (int k = 0; k < c; k++) {
                float val = img.pData[idx + k];
                val = std::max(0.f, std::min(1.f, val));
                buffer.push_back(static_cast<unsigned char>(val * 255.f));
            }
        }
    }

    return base64_encode(buffer);
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

    std::string req_str = request.dump();
    socket.send(zmq::buffer(req_str), zmq::send_flags::none);

    zmq::message_t reply;
    socket.recv(reply);

    auto response = nlohmann::json::parse(
        std::string(static_cast<char*>(reply.data()), reply.size())
    );

    // ---- EXG FEATURES ----
    int h_exg = response["shape_exg"][0];
    int w_exg = response["shape_exg"][1];
    int c_exg = response["shape_exg"][2]; // should be 104

    outFtImg_Exg.allocate(w_exg, h_exg, c_exg);

    std::vector<float> data_exg = response["data_exg"];

    for (int i = 0; i < h_exg; i++) {
        for (int j = 0; j < w_exg; j++) {
            int idx = i * w_exg + j;
            for (int k = 0; k < c_exg; k++) {
                outFtImg_Exg.pData[idx * c_exg + k] =
                    static_cast<unsigned char>(std::round(data_exg[idx * c_exg + k] * 255));
            }
        }
    }

    // ---- ELEV FEATURES ----
    int h_el = response["shape_elev"][0];
    int w_el = response["shape_elev"][1];
    int c_el = response["shape_elev"][2]; // 33

    outFtImg_Elev.allocate(w_el, h_el, c_el);

    std::vector<float> data_el = response["data_elev"];

    for (int i = 0; i < h_el; i++) {
        for (int j = 0; j < w_el; j++) {
            int idx = i * w_el + j;
            for (int k = 0; k < c_el; k++) {
                outFtImg_Elev.pData[idx * c_el + k] =
                    static_cast<unsigned char>(std::round(data_el[idx * c_el + k] * 255));
            }
        }
    }
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