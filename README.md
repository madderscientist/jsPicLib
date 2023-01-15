# jsPicLib
a small picture processor on html
## demo
[demo](https://madderscientist.github.io/jsPicLib/)

## feature
To process image on html, we use ImageData from html-canvas. However, ImageData saves all 4 channels, which sometimes is useless, especially when it's converted to grayscale image —— only one channel is needed, and three channels are same. But we still need to maintain the useless information in algorithm, which increases time and space complexity.<br>
This class imitates PIL, rearranging the data in a more clear way. It cuts channels when necessary and provides more flexible ways to process seleted channels. It's much quicker than algorithm based on ImageData when we focus on only one channel, especially grayscale image.<br>
Below is the data struct of the class:<br>
I tend to use 2D-Array to avoid 1D->2D index convert (slower but more friendly, especially in convolution algorithm). Here are two options: each pixel is an Array of all channels, or separate channels to different 2D-Arrays. I made the decision considering memory:<br>
From the website: [jsArray Memory use condition](https://www.mattzeunert.com/2016/07/24/javascript-array-object-sizes.html) we know that Arrays in Array takes more memory than numbers in Array. So we can draw a simple conclusion: less Array, less memory. As width and height are usually far bigger than channel number, of course the latter struct uses less Arrays.<br>
## DataStruct
```
jsPic.channel =
[
    [       // channel_1
        Uint8ClampedArray_1 [value_1, value_2, ...],    // line_1
        Uint8ClampedArray_1 [...],                      // line_2
        ...
    ],
    [...],  // channel_2
    ...
]
```
## API
class jsPic: no constructor
- new
- newChannel
- clone
- cloneChannel
- getPixel
- throughPic
- throughChannel
- fromImageData
- toImageData
- convert
- convert_1
- convolution
- brighten

also provided api based on ImageData:
- convert
- convolution
- gammaBright
- HSVBright

common funtction:
- rgbToHsb
- hsbToRgb

For more information please read jsPic.js<br>
Other functions like draw, zoom, paste, save... can be done with canvas api easily. The class is useful when do works like edge extraction. It's a By-product of my future project.
