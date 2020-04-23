from ffmpy import FFmpeg

if __name__ == '__main__':
    # input：输入视频，截取从00:00:06开始的10秒视频
    # output：输出图片，24fps
    ff = FFmpeg(
        inputs={"targetVideo/testVideo.mp4": '-ss 00:01:51 -t 58'},
        outputs={'videoSnapshot/image-%3d.jpg': '-r 24 -qscale:v 2 -f image2 '}
    )
    print(ff.cmd)
    ff.run()
