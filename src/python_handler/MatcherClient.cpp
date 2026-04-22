#include "MatcherClient.h"

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <vector>

// ---- simple base64 ----
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
// ------------------------

MatcherClient::MatcherClient(const std::string& address)
    : address(address) {}

std::string MatcherClient::encodeImage(const cv::Mat& img) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", img, buffer);
    return base64_encode(buffer);
}

MatchResult MatcherClient::match(const cv::Mat& img1, const cv::Mat& img2) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);

    socket.connect(address);

    nlohmann::json request;
    request["img1"] = encodeImage(img1);
    request["img2"] = encodeImage(img2);

    std::string req_str = request.dump();
    socket.send(zmq::buffer(req_str), zmq::send_flags::none);

    zmq::message_t reply;
    socket.recv(reply);

    auto response = nlohmann::json::parse(
        std::string(static_cast<char*>(reply.data()), reply.size())
    );

    MatchResult result;

    for (auto& p : response["pts1"]) {
        result.pts1.emplace_back(p[0], p[1]);
    }
    for (auto& p : response["pts2"]) {
        result.pts2.emplace_back(p[0], p[1]);
    }

    return result;
}