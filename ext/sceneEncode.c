#include <math.h>
#define HVE_IMPL
#include "hve/hve.h"
#include "scene.h"
// AV_PIX_FMT_RGB24
const int   FRAMERATE         = 60;
const char* DEVICE            = NULL;                 //NULL for default or device e.g. "/dev/dri/renderD128"
const char* ENCODER           = NULL;                 //NULL for default (h264_vaapi) or FFmpeg encoder e.g. "h264_vaapi", "hevc_vaapi", "h264_nvenc", "hevc_nvenc", ...
const char* PIXEL_FORMAT      = "nv12";               //NULL for default (NV12) or pixel format e.g. "rgb0"
const int   PROFILE           = FF_PROFILE_H264_HIGH; //or FF_PROFILE_HEVC_MAIN, FF_PROFILE_H264_CONSTRAINED_BASELINE, ...
const int   BFRAMES           = 0;                    //max_b_frames, set to 0 to minimize latency, non-zero to minimize size
const int   BITRATE           = 0;                    //average bitrate in VBR mode (bit_rate != 0 and qp == 0)
const int   QP                = 0;                    //quantization parameter in CQP mode (qp != 0 and bit_rate == 0)
const int   GOP_SIZE          = 0;                    //group of pictures size, 0 for default (determines keyframe period)
const int   COMPRESSION_LEVEL = 0;                    //encoder/codec dependent, 0 for default, for VAAPI 1-7 speed-quality tradeoff, 1 highest quality, 7 fastest
const int   VAAPI_LOW_POWER   = 0;                    //alternative VAAPI limited low-power encoding path if non-zero
const char* NVENC_PRESET      = NULL;                 //NVENC and codec specific, NULL / "" or like "default", "slow", "medium", "fast", "hp", "hq", "bd", "ll", "llhq", "llhp", "lossless", "losslesshp"
const int   NVENC_DELAY       = 0;                    //NVENC specific delay of frame output, 0 for default, -1 for 0 or positive value, set -1 to minimize latency
const int   NVENC_ZEROLATENCY = 0;                    //NVENC specific no reordering delay if non-zero, enable to minimize latency


FILE*       output_file;
struct hve* hardware_encoder;

static void encoderInit(FILE** output_file, struct hve** hardware_encoder, int width, int height) {


  struct hve_config hardware_config = {width, height, width, height, FRAMERATE,
                                       DEVICE, ENCODER, PIXEL_FORMAT, PROFILE, BFRAMES,
                                       BITRATE, QP, GOP_SIZE, COMPRESSION_LEVEL,
                                       VAAPI_LOW_POWER,
                                       NVENC_PRESET,
                                       NVENC_DELAY, NVENC_ZEROLATENCY};

  struct hve_frame frame = {0};

  //later assuming PIXEL_FORMAT is "nv12" (you may use something else)

  //fill with your stride (width including padding if any)
  frame.linesize[0] = frame.linesize[1] = width;

  AVPacket* packet; //encoded data is returned in FFmpeg packet


  //prepare file for raw H.264 output
  *output_file = fopen("output.h264", "w+b");
  if (output_file == NULL) {
    fprintf(stderr, "unable to open file for output\n");
    exit(1);
  }

  //initialize library with hve_init
  if ((*hardware_encoder = hve_init(&hardware_config)) == NULL) {
    fclose(*output_file);
    exit(1);
  }
}
static void encoderDestroy() {

  int failed; //error indicator while encoding

  hve_send_frame(hardware_encoder, NULL);
  AVPacket* packet;
  while ((packet = hve_receive_packet(hardware_encoder, &failed)))
    fwrite(packet->data, packet->size, 1, output_file);

  hve_close(hardware_encoder);
  fclose(output_file);
}
void encoderLoop(struct hve* hardware_encoder, FILE* output_file, int width, unsigned char* luminance, unsigned char* uv) {
  struct hve_frame frame = {0};

  //fill with your stride (width including padding if any)
  frame.linesize[0] = frame.linesize[1] = width;

  //encoded data is returned in FFmpeg packet
  AVPacket* packet;

  frame.data[0] = luminance;
  frame.data[1] = uv;

  //encode this frame
  if (hve_send_frame(hardware_encoder, &frame) != HVE_OK) {
    dprintf(2, "Encoder error send\n");
    exit(1);
  }

  int failed; //error indicator while encoding
  while ((packet = hve_receive_packet(hardware_encoder, &failed))) {
    //packet.data is H.264 encoded frame of packet.size length
    //here we are dumping it to raw H.264 file as example
    //yes, we ignore the return value of fwrite for simplicty
    //it could also fail in harsh real world...
    fwrite(packet->data, packet->size, 1, output_file);
  }

  //NULL packet and non-zero failed indicates failure during encoding
  if (failed) {

    dprintf(2, "Encoder error receive\n");
    exit(1);
  }
}

#include <libswscale/swscale.h>
void toLuminance(unsigned char* fbo, int width, int height, unsigned char** out_luminance, unsigned char** out_uv) {
  struct SwsContext* sws_context     = 0;
  const int          in_linesize[1]  = {3 * width};    // RGB stride
  int                out_linesize[2] = {width, width}; // NV12 stride

  uint8_t* out_planes[2];
  out_planes[0]               = malloc(width * height);
  out_planes[1]               = malloc(width * height);
  const uint8_t* in_planes[1] = {fbo};

  sws_context = sws_getCachedContext(sws_context, width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_NV12, 0, 0, 0, 0);
  sws_scale(sws_context, in_planes, in_linesize, 0, height, out_planes, out_linesize);


  *out_luminance = out_planes[0];
  *out_uv        = out_planes[1];
}

static void callback(Scene* scene, int f, const char* path) {

  unsigned char* luminance;
  unsigned char* uv;
  for (int i = 0; i < scene->desc.framesInFlight; i++) {
    unsigned char* png = scenePng(scene, i);
    toLuminance(png, scene->desc.frameBufferWidth, scene->desc.frameBufferHeight, &luminance, &uv);
    encoderLoop(hardware_encoder, output_file, scene->desc.frameBufferWidth, luminance, uv);
    free(uv);
    free(luminance);
    free(png);
  }
}
void sceneRunSuiteMovieEncode(SceneDesc sceneDesc, const char* path, void(initScene)(Scene*), void(initSceneFrame)(PushConstants* cn)) {

  encoderInit(&output_file, &hardware_encoder, sceneDesc.frameBufferWidth, sceneDesc.frameBufferHeight);
  sceneRunSuiteMovie(sceneDesc, path, initScene, initSceneFrame, callback);
  encoderDestroy();
}
