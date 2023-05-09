
/*
 * HVE Hardware Video Encoder C library header
 *
 * Copyright 2019-2023 (C) Bartosz Meglicki <meglickib@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

/**
 ******************************************************************************
 *
 *  \mainpage HVE documentation
 *  \see https://github.com/bmegli/hardware-video-encoder
 *
 *  \copyright  Copyright (C) 2019-2023 Bartosz Meglicki
 *  \file       hve.h
 *  \brief      Library public interface header
 *
 ******************************************************************************
 */

#ifndef HVE_H
#define HVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>

/** \addtogroup interface Public interface
 *  @{
 */

/**
 * @struct hve
 * @brief Internal library data passed around by the user.
 * @see hve_init, hve_close
 */
struct hve;

/**
 * @struct hve_config
 * @brief Encoder configuration
 *
 * The width and height are dimmensions of the encoded data.
 *
 * To enable hardware accelerated scaling (VAAPI only) specify non-zero
 * input_width and input_height different from width and height.
 *
 * The device can be:
 * - NULL or empty string (select automatically)
 * - point to valid device e.g. "/dev/dri/renderD128" for vaapi
 *
 * If you have multiple VAAPI devices (e.g. NVidia GPU + Intel) you may have
 * to specify Intel directly. NVidia will not work through VAAPI for encoding
 * (it works through VAAPI-VDPAU bridge and VDPAU is only for decoding).
 *
 * The encoder can be:
 * - NULL or empty string for "h264_vaapi"
 * - valid ffmpeg encoder
 *
 * You may check encoders supported by your hardware/software with ffmpeg:
 * @code
 * ffmpeg -encoders | grep vaapi
 * ffmpeg -encoders | grep nvenc
 * ffmpeg -encoders | grep h264
 * @endcode
 *
 * Encoders typically can be:
 * - h264_vaapi
 * - hevc_vaapi
 * - mjpeg_vaapi
 * - mpeg2_vaapi
 * - vp8_vaapi
 * - vp9_vaapi
 * - h264_nvenc
 * - hevc_nvenc
 * - h264_nvmpi (custom Jetson specific FFmpeg build)
 * - hevc_nvmpi (custom Jetson specific FFmpeg build)
 * - libx264 (software)
 *
 * The pixel_format (format of what you upload) typically can be:
 * - nv12 (this is generally safe choice)
 * - yuv420p (required for some encoders)
 * - yuyv422
 * - uyvy422
 * - yuv422p
 * - rgb0
 * - bgr0
 * - p010le
 *
 * You may check pixel formats supported by encoder:
 * @code
 * ffmpeg -h encoder=h264_vaapi
 * ffmpeg -h encoder=h264_nvenc
 * ffmpeg -h encoder=h264_nvmpi
 * ffmpeg -h encoder=libx264
 * @endcode
 *
 * There are no software color conversions in this library.
 *
 * For pixel format explanation see:
 * <a href="https://ffmpeg.org/doxygen/3.4/pixfmt_8h.html#a9a8e335cf3be472042bc9f0cf80cd4c5">FFmpeg pixel formats</a>
 *
 * The available profiles depend on used encoder. Use 0 to guess from input.
 *
 * For possible profiles see:
 * <a href="https://ffmpeg.org/doxygen/3.4/avcodec_8h.html#ab424d258655424e4b1690e2ab6fcfc66">FFmpeg profiles</a>
 *
 * For H.264 profile can typically be:
 * - FF_PROFILE_H264_CONSTRAINED_BASELINE
 * - FF_PROFILE_H264_MAIN
 * - FF_PROFILE_H264_HIGH
 * - ...
 *
 * For HEVC profile can typically be:
 * - FF_PROFILE_HEVC_MAIN
 * - FF_PROFILE_HEVC_MAIN_10 (10 bit channel precision)
 * - ...
 *
 * You may check profiles supported by your hardware with vainfo:
 * @code
 * vainfo --display drm --device /dev/dri/renderDXYZ
 * @endcode
 *
 * The max_b_frames controls the number of B frames.
 * Disable B frames if you need low latency (at the cost of quality/space).
 * The output will be delayed by max_b_frames+1 relative to the input.
 *
 * Set non zero bit_rate for average average bitrate and VBR mode.
 *
 * Set non zero qp for quantization parameter and CQP mode.
 *
 * If both bit_rate and qp are zero then CQP with default qp is used.
 * If both are non-zero VBR mode will be used.
 *
 * The gop_size is size of group of pictures (e.g. I, P, B frames).
 * Note that gop_size determines keyframe period.
 * Setting gop_size equal to framerate results in one keyframe per second.
 * Use 0 value for default, -1 for intra only.
 *
 * The compression_level is codec/encoder specific
 * For VAAPI it is speed-quality trade-off. Use 0 for driver default.
 * For highest quality use 1, for fastest encoding use 7.
 * The default is not highest quality so if you need it, set it explicitly to 1.
 * The exact interpretation is hardware dependent.
 *
 * The vaapi_low_power (VAAPI specific) enables alternative encoding path available on some Intel platforms.
 *
 * You may check support with vainfo (entrypoints ending with LP):
 * @code
 * vainfo --display drm --device /dev/dri/renderDXYZ
 * @endcode
 *
 * Low power encoding supports limited subset of features.
 *
 * Bitrate control with low power encoding requires loaded HuC.
 * For the details on loading HuC see:
 * <a href="https://github.com/bmegli/hardware-video-encoder/wiki/GuC-and-HuC">Loading GuC and HuC</a>
 *
 * The nvenc_preset is encoding preset to use, may be codec specific.
 *
 * The default is medium ("default", "" or NULL string)
 *
 * Typicall values: "default", "slow", "medium", "fast", "hp", "hq", "bd", "ll", "llhq", "llhp", "lossless", "losslesshp"
 *
 * You may check available presets (H.264 example)
 * @code
 * ffmpeg -h encoder=h264_nvenc -hide_banner
 * @endcode
 *
 * The nvenc_delay is delay for frame output by given amount of frames.
 * 0 leaves defaults (which is INT_MAX in FFmpeg nvenc), -1 sets 0.
 * Set to -1 (maps to 0) if you explicitly need low latency.
 *
 * The nvenc_zerolatency is NVENC specific for no reordering delay.
 * Set to non-zero if you need low latency.
 *
 * @see hve_init
 */
struct hve_config {
  int         width;             //!< width of the encoded frames
  int         height;            //!< height of the encoded frames
  int         input_width;       //!< optional scaling if non-zero and different from width
  int         input_height;      //!< optional scaling if non-zero and different from height
  int         framerate;         //!< framerate of the encoded video
  const char* device;            //!< NULL / "" or device, e.g. "/dev/dri/renderD128"
  const char* encoder;           //!< NULL / "" or encoder, e.g. "h264_vaapi"
  const char* pixel_format;      //!< NULL / "" for NV12 or format, e.g. "rgb0", "bgr0", "nv12", "yuv420p", "p010le"
  int         profile;           //!< 0 to guess from input or profile e.g. FF_PROFILE_H264_MAIN, FF_PROFILE_H264_HIGH, FF_PROFILE_HEVC_MAIN, ...
  int         max_b_frames;      //!< maximum number of B-frames between non-B-frames (disable if you need low latency)
  int         bit_rate;          //!< average bitrate in VBR mode (bit_rate != 0 and qp == 0)
  int         qp;                //!< quantization parameter in CQP mode (qp != 0 and bit_rate == 0)
  int         gop_size;          //!<  group of pictures size, 0 for default, -1 for intra only
  int         compression_level; //!< encoder/codec dependent, 0 for default, for VAAPI 1-7 speed-quality tradeoff, 1 highest quality, 7 fastest
  int         vaapi_low_power;   //!< VAAPI specific alternative limited low-power encoding if non-zero
  const char* nvenc_preset;      //!< NVENC and codec specific, NULL / "" or like "default", "slow", "medium", "fast", "hp", "hq", "bd", "ll", "llhq", "llhp", "lossless", "losslesshp"
  int         nvenc_delay;       //NVENC specific delay of frame output, 0 for default, -1 for 0 or positive value, set -1 to minimize latency
  int         nvenc_zerolatency; //NVENC specific no reordering delay if non-zero, enable to minimize latency
};

/**
 * @struct hve_frame
 * @brief Data to be encoded (single frame).
 *
 * Fill linsize array with stride (width and padding) of the data in bytes.
 * Fill data with pointers to the data (no copying is needed).
 *
 * For non planar formats only data[0] and linesize[0] is used.
 *
 * Pass the result to hve_send_frame.
 *
 * @see hve_send_frame
 */
struct hve_frame {
  uint8_t* data[AV_NUM_DATA_POINTERS];     //!< array of pointers to frame planes (e.g. Y plane and UV plane)
  int      linesize[AV_NUM_DATA_POINTERS]; //!< array of strides (width + padding) for planar frame formats
};

/**
  * @brief Constants returned by most of library functions
  */
enum hve_retval_enum {
  HVE_ERROR = -1, //!< error occured with errno set
  HVE_OK    = 0,  //!< succesfull execution
};

/**
 * @brief initialize internal library data.
 * @param config encoder configuration
 * @return
 * - pointer to internal library data
 * - NULL on error, errors printed to stderr
 *
 * @see hve_config, hve_close
 */
struct hve* hve_init(const struct hve_config* config);

/**
 * @brief free library resources
 *
 * Cleans and frees memory
 *
 * @param h pointer to internal library data
 *
 */
void hve_close(struct hve* h);

/**
 * @brief Send frame to hardware for encoding.
 *
 * Call this for each frame you want to encode.
 * Follow with hve_receive_packet to get encoded data from hardware.
 * Call with NULL frame argument to flush the encoder when you want to finish encoding.
 * After flushing follow with hve_receive_packet to get last encoded frames.
 * After flushing it is not possible to reuse the encoder.
 *
 * The pixel format of the frame should match the one specified in hve_init.
 *
 * Hardware accelerated scaling is performed before encoding if non-zero
 * input_width and input_height different from width and height were specified in hve_init.
 *
 *
 * Perfomance hints:
 *  - don't copy data from your source, just pass the pointers to data planes
 *
 * @param h pointer to internal library data
 * @param frame data to encode
 * @return
 * - HVE_OK on success
 * - HVE_ERROR indicates error
 *
 * @see hve_frame, hve_receive_packet
 *
 * Example flushing:
 * @code
 *  hve_send_frame(hardware_encoder, NULL);
 *
 *	while( (packet=hve_receive_packet(hardware_encoder, &failed)) )
 *	{
 *		//do something with packet->datag, packet->size
 *	}
 *
 *	//NULL packet and non-zero failed indicates failure during encoding
 *	if(failed)
 *		//your logic on failure
 *
 * @endcode
 *
 */
int hve_send_frame(struct hve* h, struct hve_frame* frame);


/**
 * @brief Retrieve encoded frame data from hardware.
 *
 * Keep calling this functions after hve_send_frame until NULL is returned.
 * The ownership of returned AVPacket remains with the library:
 * - consume it immidiately
 * - or copy the data
 *
 * While beginning encoding you may have to send a few frames before you will get packets.
 * When flushing the encoder you may get multiple packets afterwards.
 *
 * @param h pointer to internal library data
 * @param error pointer to error code
 * @return
 * - AVPacket * pointer to FFMpeg AVPacket, you are mainly interested in data and size members
 * - NULL when no more data is pending, query error parameter to check result (HVE_OK on success)
 *
 * @see hve_send_frame
 *
 * Example (in encoding loop):
 * @code
 *  if( hve_send_frame(hardware_encoder, &frame) != HVE_OK)
 *		break; //break on error
 *
 *	while( (packet=hve_receive_packet(hardware_encoder, &failed)) )
 *	{
 *		//do something with packet->data, packet->size
 *	}
 *
 *	//NULL packet and non-zero failed indicates failure during encoding
 *	if(failed)
 *		break; //break on error
 * @endcode
 *
 */
AVPacket* hve_receive_packet(struct hve* h, int* error);

/** @}*/


#ifdef HVE_IMPL
// FFmpeg
#  include <libavcodec/avcodec.h>
#  include <libavutil/hwcontext.h>
#  include <libavutil/pixdesc.h>
#  include <libavfilter/buffersink.h>
#  include <libavfilter/buffersrc.h>

#  include <stdio.h>  //fprintf
#  include <stdlib.h> //malloc
#  include <string.h> //strstr

// internal library data passed around by the user
struct hve {
  enum AVPixelFormat sw_pix_fmt;
  AVBufferRef*       hw_device_ctx;
  AVCodecContext*    avctx;

  //accelerated scaling related
  AVFilterContext* buffersrc_ctx;
  AVFilterContext* buffersink_ctx;
  AVFilterGraph*   filter_graph;

  AVFrame* sw_frame; //software
  AVFrame* hw_frame; //hardware
  AVFrame* fr_frame; //filter
  AVPacket enc_pkt;
};

static struct hve* hve_close_and_return_null(struct hve* h, const char* msg);

static int init_hwframes_context(struct hve* h, const struct hve_config* config, enum AVHWDeviceType device_type);
static int init_hardware_scaling(struct hve* h, const struct hve_config* config);

static enum AVHWDeviceType hve_hw_device_type(const char* encoder);
static enum AVPixelFormat  hve_hw_pixel_format(enum AVHWDeviceType type);
static int                 hve_pixel_format_depth(enum AVPixelFormat pix_fmt, int* depth);

static int HVE_ERROR_MSG(const char* msg);
static int HVE_ERROR_MSG_FILTER(AVFilterInOut* ins, AVFilterInOut* outs, const char* msg);

static int hw_upload(struct hve* h);
static int scale_encode(struct hve* h);
static int encode(struct hve* h);

// NULL on error
struct hve* hve_init(const struct hve_config* config) {
  struct hve *h, zero_hve = {0};
  int         err;
  AVCodec*    codec = NULL;

  if ((h = (struct hve*)malloc(sizeof(struct hve))) == NULL)
    return hve_close_and_return_null(NULL, "not enough memory for hve");

  *h = zero_hve; //set all members of dynamically allocated struct to 0 in a portable way

  /*
	avcodec_register_all(); // for compatibility with FFmpeg 3.4 (e.g. Ubuntu 18.04)
	avfilter_register_all();// for compatibility with FFmpeg 3.4 (e.g. Ubuntu 18.04) 
  */
  av_log_set_level(AV_LOG_VERBOSE);

  //specified encoder or NULL / empty string for H.264 VAAPI
  const char* encoder = (config->encoder != NULL && config->encoder[0] != '\0') ? config->encoder : "h264_vaapi";

  enum AVHWDeviceType device_type = hve_hw_device_type(encoder);

  if (device_type == AV_HWDEVICE_TYPE_NONE)
    fprintf(stderr, "hve: not using hardware device type (enoder wrapper, software or hardware not supported by hve)\n");

  if (!(codec = avcodec_find_encoder_by_name(encoder)))
    return hve_close_and_return_null(h, "could not find encoder");

  if (!(h->avctx = avcodec_alloc_context3(codec)))
    return hve_close_and_return_null(h, "unable to alloc codec context");

  h->avctx->width  = config->width;
  h->avctx->height = config->height;

  if (config->gop_size) //0 for default, -1 for intra only
    h->avctx->gop_size = (config->gop_size != -1) ? config->gop_size : 0;

  h->avctx->time_base           = (AVRational){1, config->framerate};
  h->avctx->framerate           = (AVRational){config->framerate, 1};
  h->avctx->sample_aspect_ratio = (AVRational){1, 1};

  if (config->profile)
    h->avctx->profile = config->profile;

  h->avctx->max_b_frames = config->max_b_frames;
  h->avctx->bit_rate     = config->bit_rate;

  if (config->compression_level)
    h->avctx->compression_level = config->compression_level;

  //try to find software pixel format that user wants to upload data in
  if (config->pixel_format == NULL || config->pixel_format[0] == '\0')
    h->sw_pix_fmt = AV_PIX_FMT_NV12;
  else if ((h->sw_pix_fmt = av_get_pix_fmt(config->pixel_format)) == AV_PIX_FMT_NONE) {
    fprintf(stderr, "hve: failed to find pixel format %s\n", config->pixel_format);
    return hve_close_and_return_null(h, NULL);
  }

  h->avctx->pix_fmt = h->sw_pix_fmt;

  if (device_type != AV_HWDEVICE_TYPE_NONE)
    if ((err = init_hwframes_context(h, config, device_type)) < 0)
      return hve_close_and_return_null(h, "failed to set hwframe context");

  AVDictionary* opts = NULL;

  if (config->qp && (av_dict_set_int(&opts, "qp", config->qp, 0) < 0))
    return hve_close_and_return_null(h, "failed to initialize option dictionary (qp)");

  if (config->vaapi_low_power && (av_dict_set_int(&opts, "low_power", config->vaapi_low_power != 0, 0) < 0))
    return hve_close_and_return_null(h, "failed to initialize option dictionary (low_power)");

  if (config->nvenc_preset && config->nvenc_preset[0] != '\0' && (av_dict_set(&opts, "preset", config->nvenc_preset, 0) < 0))
    return hve_close_and_return_null(h, "failed to initialize option dictionary (NVENC preset)");

  if (config->nvenc_delay && (av_dict_set_int(&opts, "delay", (config->nvenc_delay > 0) ? config->nvenc_delay : 0, 0) < 0))
    return hve_close_and_return_null(h, "failed to initialize option dictionary (NVENC delay)");

  if (config->nvenc_zerolatency && (av_dict_set_int(&opts, "zerolatency", config->nvenc_zerolatency != 0, 0) < 0))
    return hve_close_and_return_null(h, "failed to initialize option dictionary (NVENC zerolatency)");

  if ((err = avcodec_open2(h->avctx, codec, &opts)) < 0) {
    av_dict_free(&opts);
    return hve_close_and_return_null(h, "cannot open video encoder codec");
  }

  AVDictionaryEntry* de = NULL;

  while ((de = av_dict_get(opts, "", de, AV_DICT_IGNORE_SUFFIX)))
    fprintf(stderr, "hve: %s codec option not found\n", de->key);

  av_dict_free(&opts);

  if ((config->input_width && config->input_width != config->width) ||
      (config->input_height && config->input_height != config->height))
    if (init_hardware_scaling(h, config) < 0)
      return hve_close_and_return_null(h, "failed to initialize hardware scaling");
  //from now on h->filter_graph may be used to check if scaling was requested
  if (h->filter_graph)
    if (!(h->fr_frame = av_frame_alloc()))
      return hve_close_and_return_null(h, "av_frame_alloc not enough memory (filter frame)");

  if (!(h->sw_frame = av_frame_alloc()))
    return hve_close_and_return_null(h, "av_frame_alloc not enough memory (software frame");

  h->sw_frame->width  = config->input_width ? config->input_width : config->width;
  h->sw_frame->height = config->input_height ? config->input_height : config->height;
  h->sw_frame->format = h->sw_pix_fmt;

  av_init_packet(&h->enc_pkt);
  h->enc_pkt.data = NULL;
  h->enc_pkt.size = 0;

  return h;
}

void hve_close(struct hve* h) {
  if (h == NULL)
    return;

  av_packet_unref(&h->enc_pkt);
  av_frame_free(&h->sw_frame);
  av_frame_free(&h->fr_frame);
  av_frame_free(&h->hw_frame);

  avfilter_graph_free(&h->filter_graph);

  avcodec_free_context(&h->avctx);
  av_buffer_unref(&h->hw_device_ctx);

  free(h);
}

static struct hve* hve_close_and_return_null(struct hve* h, const char* msg) {
  if (msg)
    fprintf(stderr, "hve: %s\n", msg);

  hve_close(h);

  return NULL;
}

static int init_hwframes_context(struct hve* h, const struct hve_config* config, enum AVHWDeviceType device_type) {
  AVBufferRef*       hw_frames_ref;
  AVHWFramesContext* frames_ctx = NULL;
  int                err        = 0, depth;

  //specified device or NULL / empty string for default
  const char* device = (config->device != NULL && config->device[0] != '\0') ? config->device : NULL;

  if ((h->avctx->pix_fmt = hve_hw_pixel_format(device_type)) == AV_PIX_FMT_NONE)
    return HVE_ERROR_MSG("could not find hardware pixel format for encoder");

  if (av_hwdevice_ctx_create(&h->hw_device_ctx, device_type, device, NULL, 0) < 0)
    return HVE_ERROR_MSG("failed to create hardware device context");

  if (!(hw_frames_ref = av_hwframe_ctx_alloc(h->hw_device_ctx)))
    return HVE_ERROR_MSG("failed to create hardware frame context");

  frames_ctx         = (AVHWFramesContext*)(hw_frames_ref->data);
  frames_ctx->format = h->avctx->pix_fmt; //e.g. AV_PIX_FMT_VAAPI, AV_PIX_FMT_CUDA

  frames_ctx->width  = config->input_width ? config->input_width : config->width;
  frames_ctx->height = config->input_height ? config->input_height : config->height;

  frames_ctx->initial_pool_size = 20;

  frames_ctx->sw_format = h->sw_pix_fmt;

  // Starting from FFmpeg 4.1, avcodec will not fall back to NV12 automatically
  // when using non 4:2:0 software pixel format not supported by codec with VAAPI.
  // Here, instead of using h->sw_pix_fmt we always fall to P010LE for 10 bit
  // input and NV12 otherwise which may possibly lead to some loss of information
  // on modern hardware supporting 4:2:2 and 4:4:4 chroma subsampling
  // (e.g. HEVC with >= IceLake)
  // See:
  // https://github.com/bmegli/hardware-video-encoder/issues/26
  // https://github.com/bmegli/hardware-video-encoder/issues/35
  if (frames_ctx->format == AV_PIX_FMT_VAAPI) {
    frames_ctx->sw_format = AV_PIX_FMT_NV12;

    if (hve_pixel_format_depth(h->sw_pix_fmt, &depth) != HVE_OK)
      return HVE_ERROR_MSG("failed to get pixel format depth");

    if (depth == 10)
      frames_ctx->sw_format = AV_PIX_FMT_P010LE;
  }

  if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {
    fprintf(stderr, "hve: failed to initialize hardware frame context - \"%s\"\n", av_err2str(err));
    av_buffer_unref(&hw_frames_ref);
    return HVE_ERROR_MSG("hint - make sure you are using supported pixel format");
  }

  h->avctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
  if (!h->avctx->hw_frames_ctx)
    err = AVERROR(ENOMEM);

  av_buffer_unref(&hw_frames_ref);
  return err == 0 ? HVE_OK : HVE_ERROR;
}

static int init_hardware_scaling(struct hve* h, const struct hve_config* config) {
  const AVFilter *buffersrc, *buffersink;
  AVFilterInOut * ins, *outs;
  char            temp_str[128];
  int             err = 0;

  if (!(buffersrc = avfilter_get_by_name("buffer")))
    return HVE_ERROR_MSG("unable to find filter 'buffer'");

  if (!(buffersink = avfilter_get_by_name("buffersink")))
    return HVE_ERROR_MSG("unable to find filter 'buffersink'");

  //allocate memory
  ins             = avfilter_inout_alloc();
  outs            = avfilter_inout_alloc();
  h->filter_graph = avfilter_graph_alloc(); //has to be fred with HVE cleanup

  if (!ins || !outs || !h->filter_graph)
    return HVE_ERROR_MSG_FILTER(ins, outs, "unable to allocate memory for the filter");

  //prepare filter source
  snprintf(temp_str, sizeof(temp_str), "video_size=%dx%d:pix_fmt=%d:time_base=1/%d:pixel_aspect=1/1",
           config->input_width, config->input_height, AV_PIX_FMT_VAAPI, config->framerate);

  if (avfilter_graph_create_filter(&h->buffersrc_ctx, buffersrc, "in", temp_str, NULL, h->filter_graph) < 0)
    return HVE_ERROR_MSG_FILTER(ins, outs, "cannot create buffer source");

  outs->name       = av_strdup("in");
  outs->filter_ctx = h->buffersrc_ctx;
  outs->pad_idx    = 0;
  outs->next       = NULL;

  //initialize buffersrc with hw frames context
  AVBufferSrcParameters* par;

  if (!(par = av_buffersrc_parameters_alloc()))
    return HVE_ERROR_MSG_FILTER(ins, outs, "unable to allocate memory for the filter (params)");

  par->hw_frames_ctx = h->avctx->hw_frames_ctx;

  err = av_buffersrc_parameters_set(h->buffersrc_ctx, par);
  av_free(par);
  if (err < 0)
    return HVE_ERROR_MSG_FILTER(ins, outs, "unable to initialize buffersrc with hw frames context");

  //prepare filter sink
  if (avfilter_graph_create_filter(&h->buffersink_ctx, buffersink, "out", NULL, NULL, h->filter_graph) < 0)
    return HVE_ERROR_MSG_FILTER(ins, outs, "cannot create buffer sink");

  ins->name       = av_strdup("out");
  ins->filter_ctx = h->buffersink_ctx;
  ins->pad_idx    = 0;
  ins->next       = NULL;

  //the actual description of the graph
  snprintf(temp_str, sizeof(temp_str), "format=vaapi,scale_vaapi=w=%d:h=%d", config->width, config->height);

  if (avfilter_graph_parse_ptr(h->filter_graph, temp_str, &ins, &outs, NULL) < 0)
    return HVE_ERROR_MSG_FILTER(ins, outs, "failed to parse filter graph description");

  for (unsigned i = 0; i < h->filter_graph->nb_filters; i++)
    if (!(h->filter_graph->filters[i]->hw_device_ctx = av_buffer_ref(h->hw_device_ctx)))
      return HVE_ERROR_MSG_FILTER(ins, outs, "not enough memory to reference hw device ctx by filters");

  if (avfilter_graph_config(h->filter_graph, NULL) < 0)
    return HVE_ERROR_MSG_FILTER(ins, outs, "failed to configure filter graph");

  avfilter_inout_free(&ins);
  avfilter_inout_free(&outs);

  return HVE_OK;
}

static enum AVHWDeviceType hve_hw_device_type(const char* encoder) {
  if (strstr(encoder, "vaapi"))
    return AV_HWDEVICE_TYPE_VAAPI;
  else if (strstr(encoder, "nvenc"))
    return AV_HWDEVICE_TYPE_CUDA;

  return AV_HWDEVICE_TYPE_NONE;
}

static enum AVPixelFormat hve_hw_pixel_format(enum AVHWDeviceType type) {
  if (type == AV_HWDEVICE_TYPE_VAAPI)
    return AV_PIX_FMT_VAAPI;
  else if (type == AV_HWDEVICE_TYPE_CUDA)
    return AV_PIX_FMT_CUDA;

  return AV_PIX_FMT_NONE;
}

static int hve_pixel_format_depth(enum AVPixelFormat pix_fmt, int* depth) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pix_fmt);
  int                       i;

  if (!desc || !desc->nb_components)
    return HVE_ERROR;

  *depth = -INT_MAX;

  for (i = 0; i < desc->nb_components; i++)
    *depth = FFMAX(desc->comp[i].depth, *depth);

  return HVE_OK;
}

static int HVE_ERROR_MSG(const char* msg) {
  fprintf(stderr, "hve: %s\n", msg);
  return HVE_ERROR;
}

static int HVE_ERROR_MSG_FILTER(AVFilterInOut* ins, AVFilterInOut* outs, const char* msg) {
  avfilter_inout_free(&ins);
  avfilter_inout_free(&outs);
  //h->filter_graph is fred in reaction to init_hardware_scaling HVE_ERROR return

  return HVE_ERROR_MSG(msg);
}

int hve_send_frame(struct hve* h, struct hve_frame* frame) {
  //note - in case hardware frame preparation fails, the frame is fred:
  // - here (this is next user try)
  // - or in av_close (this is user decision to terminate)
  av_frame_free(&h->hw_frame);

  // NULL frame is used for flushing the encoder
  if (frame == NULL) {
    if (h->filter_graph)
      if (av_buffersrc_add_frame_flags(h->buffersrc_ctx, NULL, AV_BUFFERSRC_FLAG_KEEP_REF | AV_BUFFERSRC_FLAG_PUSH))
        fprintf(stderr, "hve: error while marking filter EOF\n");

    if (avcodec_send_frame(h->avctx, NULL) < 0)
      return HVE_ERROR_MSG("error while flushing encoder");

    return HVE_OK;
  }

  //this just copies a few ints and pointers, not the actual frame data
  memcpy(h->sw_frame->linesize, frame->linesize, sizeof(frame->linesize));
  memcpy(h->sw_frame->data, frame->data, sizeof(frame->data));

  if (h->hw_device_ctx)
    if (hw_upload(h) < 0)
      return HVE_ERROR_MSG("failed to upload frame data to hardware");

  if (h->filter_graph)
    return scale_encode(h);

  return encode(h);
}

static int hw_upload(struct hve* h) {
  if (!(h->hw_frame = av_frame_alloc()))
    return HVE_ERROR_MSG("av_frame_alloc not enough memory for hw_frame");

  if (av_hwframe_get_buffer(h->avctx->hw_frames_ctx, h->hw_frame, 0) < 0)
    return HVE_ERROR_MSG("av_hwframe_get_buffer error");

  if (!h->hw_frame->hw_frames_ctx)
    return HVE_ERROR_MSG("hw_frame->hw_frames_ctx not enough memory");

  if (av_hwframe_transfer_data(h->hw_frame, h->sw_frame, 0) < 0)
    return HVE_ERROR_MSG("error while transferring frame data to surface");

  return HVE_OK;
}

static int scale_encode(struct hve* h) {
  int err, err2;

  if (av_buffersrc_add_frame_flags(h->buffersrc_ctx, h->hw_frame, AV_BUFFERSRC_FLAG_KEEP_REF | AV_BUFFERSRC_FLAG_PUSH) < 0)
    return HVE_ERROR_MSG("failed to push frame to filtergraph");

  while ((err = av_buffersink_get_frame(h->buffersink_ctx, h->fr_frame)) >= 0) {
    err2 = avcodec_send_frame(h->avctx, h->fr_frame);
    av_frame_unref(h->fr_frame);

    if (err2 < 0)
      return HVE_ERROR_MSG("send_frame error (after scaling)");
  }

  if (err == AVERROR(EAGAIN) || err == AVERROR_EOF)
    return HVE_OK;

  if (err < 0)
    return HVE_ERROR_MSG("failed to get frame from filtergraph");

  return HVE_OK;
}

static int encode(struct hve* h) {
  AVFrame* frame = h->hw_frame ? h->hw_frame : h->sw_frame;

  if (avcodec_send_frame(h->avctx, frame) < 0)
    return HVE_ERROR_MSG("send_frame error");

  return HVE_OK;
}

// returns:
// - non NULL on success
// - NULL and failed == false if more data is needed
// - NULL and failed == true on error
// the ownership of returned AVPacket* remains with the library
AVPacket* hve_receive_packet(struct hve* h, int* error) {
  //the packed will be unreffed in:
  //- next call to av_receive_packet through avcodec_receive_packet
  //- av_close (user decides to finish in the middle of encoding)
  //whichever happens first
  int ret = avcodec_receive_packet(h->avctx, &h->enc_pkt);

  *error = HVE_OK;

  if (ret == 0)
    return &h->enc_pkt;

  //EAGAIN means that we need to supply more data
  //EOF means that we are flushing the decoder and no more data is pending
  //otherwise we got an error
  *error = (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ? HVE_OK : HVE_ERROR;
  return NULL;
}


#endif
#ifdef __cplusplus
}
#endif

#endif
