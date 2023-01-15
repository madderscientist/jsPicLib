/**
 * @description a picture tool
 * @author madderscientist.github
 * @DataStruct
 *  channel [
 *      channel_1 [
 *          Uint8ClampedArray_1 [value_1, value_2, ...],
 *          Uint8ClampedArray_2 [...],
 *          ...
 *      ],
 *      channel_2 [...],
 *      ...
 *  ]
 * @example
 *  let j = new jsPic().fromImageData(imagedata,'L')
 *  j = j.convolution({kernel:jsPic.Laplacian, picfun:(x=>Math.abs(x)>200?255:0), padding:[1,1]})
 */
class jsPic {
    // useful kernels
    static Gaussian = [[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]
    static Prewitt_H = [[-1,0,1],[-1,0,1],[-1,0,1]]
    static Prewitt_V = [[-1,-1,-1],[0,0,0],[1,1,1]]
    static Sobel_H = [[-1,0,1],[-2,0,2],[-1,0,1]]
    static Sobel_V = [[-1,-2,-1],[0,0,0],[1,2,1]]
    static Laplacian = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

    /**
     * new a jsPic from param
     * @param {number} width the picture's width
     * @param {number} height the picture's height
     * @param {number | Array} channel if number, initialize data filled with param 'fill'; if Array, use it directly
     * @param {Int8Array} fill fill[i] = channel{i}'s default value (only used when 'channel' is a number)
     * @returns {jsPic} overwirte and return itself
     */
    new(width, height, channel = 4, fill = [255,255,255,255]) {
        this.width = width;
        this.height = height;
        if (Array.isArray(channel)) this.channel = channel;
        else this.channel = Array.from({length:channel},(_,i)=>{this.newChannel(fill[i]);});
        return this;
    }

    /**
     * new a channel filled with 'fill'
     * @param {number} fill default value
     * @returns {Array} a channel
     */
    newChannel(fill) {
        let Channel = new Array(this.height);
        for (let h = 0; h < this.height; h++)
            Channel[h] = new Uint8ClampedArray(this.width).fill(fill);
        return Channel;
    }

    /**
     * deeply copy a channel
     * @param {number} channel target channel index
     * @returns {Array} copy
     */
    cloneChannel(channel) {return Array.from(this.channel[channel], (line) => new Uint8ClampedArray(line));}

    /**
     * deeply clone a jsPic
     * @returns {jsPic} copy
     */
    clone() {return new jsPic().new(this.width, this.height, Array.from(this.channel, (_, index) => this.cloneChannel(index)));}
    
    /**
     * get a pixel at (x,y)
     * @param {number} x
     * @param {number} y
     * @returns {Array} [channel_1, channel_2, ... ]
     */
    getPixel(x, y) {
        p = new Uint8ClampedArray(this.channel.length);
        for (let i = 0; i < this.channel.length; i++) p[i] = this.channel[i][y][x];
        return p;
    }

    /**
     * traverse this.channel[channel], replacing each value with mapfn(value)
     * @param {number} channel target channel index
     * @param {Function} mapfn input: each value; its output will be used
     */
    throughChannel(channel, mapfn = x => x) {
        let d = this.channel[channel];
        for (let h = 0; h < this.height; h++)
            for (let w = 0; w < this.width; w++)
                d[h][w] = mapfn(d[h][w]);
    }

    /**
     * traverse all channels together, replacing each pixel with mapfn(pixel)
     * @param {Function} pixfun input: [channel_1, channel_2, ... ]; its output will be distributed to each channel
     * @returns {jsPic} change and return itself
     */
    throughPic(pixfun = x => { return x; }) {
        for (let h = 0; h < this.height; h++) {
            for (let w = 0; w < this.width; w++) {
                let after = pixfun(this.getPixel(w, h));
                for (let c = 0; c < this.channel.length; c++)
                    this.channel[c][h][w] = after[c];
            }
        } return this;
    }

    /**
     * construct jsPic from ImageData
     * @param {ImageData} ImgData ImageData from html canvas
     * @param {String} mode convert mode (convert here is quicker than convert afterwards)
     * @param {number} threshold only used when mode = '1'
     * @returns {jsPic} overwrite and return itself
     */
    fromImageData(ImgData, mode = 'RGBA', threshold = 127) {
        this.height = ImgData.height;
        this.width = ImgData.width;
        let d = ImgData.data;
        switch (mode) {
            case 'RGBA': case 'RGB': {
                let l = mode.length;
                this.channel = new Array(l);
                for (let c = 0; c < l; c++) {
                    let Channel = new Array(this.height);
                    for (let h = 0, k = c; h < this.height; h++) {
                        let Line = new Uint8ClampedArray(this.width);
                        for (let w = 0; w < this.width; w++, k += 4) Line[w] = d[k];
                        Channel[h] = Line;
                    } this.channel[c] = Channel;
                } break;
            }
            case 'L': {
                let Channel = new Array(this.height);
                for (let h = 0, k = 0; h < this.height; h++) {
                    let Line = new Uint8ClampedArray(this.width);
                    for (let w = 0; w < this.width; w++, k += 4)
                        Line[w] = d[k] * 0.299 + d[k + 1] * 0.587 + d[k + 2] * 0.114;
                    Channel[h] = Line;
                } this.channel = [Channel];
                break;
            }
            case '1': {
                let Channel = new Array(this.height);
                for (let h = 0, k = 0; h < this.height; h++) {
                    let Line = new Uint8ClampedArray(this.width);
                    for (let w = 0; w < this.width; w++, k += 4)
                        Line[w] = d[k] * 0.299 + d[k + 1] * 0.587 + d[k + 2] * 0.114 > threshold ? 255 : 0;
                    Channel[h] = Line;
                } this.channel = [Channel];
                break;
            }
        }
        return this;
    }
    
    /**
     * construct ImageData from jsPic
     * @param {Array} select select 4 channels to form ImageData, ImageData's channel[i] = this.channel[select[i]]
     * @param {Array} fill use channel filled with fill[i] when select[i] is illegal
     * @returns {ImageData}
     */
    toImageData(select = [0, 1, 2, 3], fill = [255, 255, 255, 255]) {
        if (select.length != 4 || fill.length != 4) {
            console.error("index error!"); return null;
        }
        let data = new Uint8ClampedArray(4 * this.width * this.height);
        let k = 0;
        for (let h = 0; h < this.height; h++) {
            for (let w = 0; w < this.width; w++) {
                for (let c = 0; c < 4; c++) {
                    // 如果select[c]不合法则用fill填充
                    if (select[c] >= this.channel.length || select[c] < 0) data[k++] = fill[c];
                    else data[k++] = this.channel[select[c]][h][w];
                }
            }
        }
        return new ImageData(data, this.width, this.height);
    }

    /**
     * mode convert without changing itself
     * @param {String} mode 
     * @param {number} threshold only used when mode='1'
     * @returns {jsPic} new jsPic
     */
    convert(mode = 'L', threshold = 127) {
        switch (mode) {
            case 'L':       // gray
                if (this.channel.length < 3) break;
                let C = new Array(this.height);
                for (let h = 0; h < this.height; h++) {
                    let L = new Uint8ClampedArray(this.width);
                    for (let w = 0; w < this.width; w++) {
                        L[w] = this.channel[0][h][w] * 0.299 + this.channel[1][h][w] * 0.587 + this.channel[2][h][w] * 0.114;
                    }
                    C[h] = L;
                }
                return new jsPic().new(this.width, this.height, [C]);
            case '1':       // black & white
                if (this.channel.length >= 3) {
                    let L = this.convert('L');
                    L.convert_1(0, threshold);
                    return L;
                } else if (this.channel.length == 1) {
                    let C = new Array(this.height);
                    for (let h = 0; h < this.height; h++) {
                        let L = new Uint8ClampedArray(this.width);
                        for (let w = 0; w < this.width; w++) {
                            L[w] = this.channel[0][h][w] > threshold ? 255 : 0;
                        }
                        C[h] = L;
                    }
                    return new jsPic().new(this.width, this.height, [C]);
                } else break;
            case 'RGB':
                if (this.channel.length == 3) return this.clone();
                else if (this.channel.length == 4)
                    return new jsPic().new(this.width, this.height, Array.from({ length: this.channel.length - 1 }, (_, index) => this.cloneChannel(index)));
                else break;
            default: console.error("unknown mode!"); return null;
        }
        console.error("channel number error!"); return null;
    }

    /**
     * Binarization one channel (change itself)
     * @param {number} channel target channel index
     * @param {*} threshold 
     */
    convert_1(channel, threshold = 127) {this.throughChannel(channel, (x) => { return x > threshold ? 255 : 0; });}

    /**
     * convoluion on selected channels
     * @param {Object} setting {kernel:Array(n*n), stride:Array(2), padding:Array(2), fill:Array(channel.length), pixfun:Function, select:Array} 
     * @returns new jsPic
     */
    convolution({ kernel, stride = [1, 1], padding = [0, 0], fill = [127, 127, 127, 0], pixfun = x => { return x; }, select = [0, 1, 2] }) {
        let kernelW = kernel[0].length;
        let kernelH = kernel.length;
        let newHeight = Math.floor((this.height + 2 * padding[1] - kernelH) / stride[1] + 1);
        let newWidth = Math.floor((this.width + 2 * padding[0] - kernelW) / stride[0] + 1);
        let C = new Array(this.channel.length);
        for (let c = 0; c < this.channel.length; c++) {
            let i = select.indexOf(c);
            if (i != -1) {
                let cha = new Array(newHeight);
                for (let h = -padding[1], H = 0; h < this.height + padding[1] - kernelH + 1; h += stride[1], H++) {
                    let L = new Uint8ClampedArray(newWidth);
                    for (let w = -padding[0], W = 0; w < this.width + padding[0] - kernelW + 1; w += stride[0], W++) {
                        // 遍历卷积核
                        let sum = 0;
                        for (let hh = 0; hh < kernelH; hh++) {
                            for (let ww = 0; ww < kernelW; ww++) {
                                let kh = h + hh;
                                let kw = w + ww;
                                let value = (kw < 0 || kw >= this.width || kh < 0 || kh >= this.height) ? fill[i] : this.channel[c][kh][kw];
                                sum += value * kernel[hh][ww];
                            }
                        } L[W] = pixfun(sum);
                    } cha[H] = L;
                } C[c] = cha;
            } else C[c] = this.cloneChannel(c);
        }
        return new jsPic().new(newWidth, newHeight, C);
    }

    /**
     * change selected channels' brightness
     * @param {String} mode 'gamma' | 'hsb' | 'linear'
     * @param {number} extent >0 meaning changes with mode
     * @param {Array} channels target channels' index
     * @returns {jsPic | null} change and return itself when succeeded. otherwise null
     */
    brighten(mode = 'gamma', extent = 1, channels = [0,1,2]) {
        switch(mode){
            case 'gamma':
                extent = 1/extent;
                let gammaMap = Array.from({length:256},(_,i)=>255 * Math.pow(i / 255, extent));
                for(let i=0;i<channels.length;i++) this.throughChannel(i,x=>gammaMap[x]);
                return this;
            case 'hsb':
                if(this.channel.length<3||channels.length!=3) break;
                return this.throughPic((pixel)=>{
                    let hsb = rgbToHsb([pixel[channels[0]],pixel[channels[1]],pixel[channels[2]]]);
                    hsb[2] = Math.min(100, hsb[2] * x);
                    let result = hsbToRgb(hsb);
                    for(let c=0;c<3;c++) pixel[channels[c]] = result[c];
                    return pixel;
                });
            case 'linear':
                for(let c=0;c<channels.length;c++){
                    this.throughChannel(channels[c],x=>extent*x);
                    return this;
                }
            default: console.error("unknown mode!"); return null;
        }
        console.error("channel number error!"); return null;
    }
}

function hsbToRgb(hsb) {
    let h = hsb[0], s = hsb[1] / 100, v = hsb[2] / 100;
    let r = 0, g = 0, b = 0;
    let i = parseInt((h / 60) % 6);
    let f = h / 60 - i;
    let p = v * (1 - s);
    let q = v * (1 - f * s);
    let t = v * (1 - (1 - f) * s);
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return [parseInt(r * 255), parseInt(g * 255), parseInt(b * 255)];
}

function rgbToHsb(rgb) {
    let h = 0, s = 0, v = 0;
    let r = rgb[0], g = rgb[1], b = rgb[2];
    arr.sort((a, b) => { return a - b; });
    let max = arr[2];
    let min = arr[0];
    v = max / 255;
    if (max === 0) s = 0;
    else s = 1 - (min / max);
    if (max === min) h = 0;     // max===min的时候，h无论为多少都无所谓
    else if (max === r && g >= b) h = 60 * ((g - b) / (max - min)) + 0;
    else if (max === r && g < b) h = 60 * ((g - b) / (max - min)) + 360;
    else if (max === g) h = 60 * ((b - r) / (max - min)) + 120;
    else if (max === b) h = 60 * ((r - g) / (max - min)) + 240;
    return [parseInt(h), parseInt(s * 100), parseInt(v * 100)];
}

/*===== below are functions execute directly on ImageData =====*/
function convert(ImgData, mode = 'L', threshold = 127) {
    var d = ImgData.data;
    switch (mode) {
        case 'L':       // 转灰度
            for (let i = 0; i < d.length; i += 4) {
                Gray = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
                d[i] = Gray;
                d[i + 1] = Gray;
                d[i + 2] = Gray;
            }
            break;
        case '1':       // 转黑白
            for (let i = 0; i < d.length; i += 4) {
                Gray = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
                Gray = (Gray > threshold) ? 255 : 0;
                d[i] = Gray;
                d[i + 1] = Gray;
                d[i + 2] = Gray;
            }
            break;
    }
}


function convolution(ImgData, kernel, stride = 1, padding = 0, fill = [127, 127, 127, 0], pixfun = x=>{return x;}) {
    let xStep, yStep
    if (Array.isArray(stride)) {
        xStep = stride[0];
        yStep = stride[1];
    } else {
        xStep = yStep = stride;
    } stride = null;

    let xPad, yPad;
    if (Array.isArray(padding)) {
        xPad = padding[0];
        yPad = padding[1];
    } else {
        xPad = yPad = padding;
    } padding = null;

    let kernelW = kernel[0].length;
    let kernelH = kernel.length;
    let newHeight = Math.floor((ImgData.height + 2 * yPad - kernelH) / yStep + 1)
    let newWidth = Math.floor((ImgData.width + 2 * xPad - kernelW) / yStep + 1)
    let newD = new Uint8ClampedArray(4 * newHeight * newWidth);

    d = ImgData.data;
    function xyToIndex(x, y) {
        if (x < 0 || x >= ImgData.width || y < 0 || y >= ImgData.height) return fill;
        else {
            let i = 4 * (y * ImgData.width + x);
            return [d[i], d[i + 1], d[i + 2], d[i + 3]]
        }
    }

    // 遍历图片
    let indexD = 0;
    for (let h = -yPad; h < ImgData.height + yPad - kernelH + 1; h += yStep) {
        for (let w = -xPad; w < ImgData.width + xPad - kernelW + 1; w += xStep) {
            let pix = [0, 0, 0]
            // 遍历卷积核
            for (let hh = 0; hh < kernelH; hh++) {
                for (let ww = 0; ww < kernelW; ww++) {
                    let pixel = xyToIndex(w + ww, h + hh);
                    // 遍历三个通道
                    for (let c = 0; c < 3; c++) pix[c] += kernel[hh][ww] * pixel[c]
                }
            }
            for (let i = 0; i < 3; i++) newD[indexD + i] = pixfun(pix[i])
            newD[indexD + 3] = 255;     // 不透明度不参与计算
            indexD += 4;
        }
    }
    return new ImageData(newD, newWidth, newHeight);
}


function gammaBright(ImgData, g) {      // windows照片编辑为此法
    gammaMap = new Uint8ClampedArray(256);
    if (g > 0) { for (let i = 0; i < 256; i++) gammaMap[i] = 255 * Math.pow(i / 255, 1 / g) }
    else if (g == 0) { for (let i = 0; i < 256; i++) gammaMap[i] = 0; }
    else return;
    d = ImgData.data
    for (let i = 0; i < d.length; i += 4) {
        d[i] = gammaMap[d[i]];
        d[i + 1] = gammaMap[d[i + 1]];
        d[i + 2] = gammaMap[d[i + 2]];
    }
}

function HSVBright(ImgData, x) {    // PS&PIL 效果类似此法
    d = ImgData.data;
    for (let i = 0; i < d.length; i += 4) {
        hsb = rgbToHsb([d[i], d[i + 1], d[i + 2]]);
        hsb[2] = Math.min(100, hsb[2] * x)
        rgb = hsbToRgb(hsb)
        d[i] = rgb[0]
        d[i + 1] = rgb[1]
        d[i + 2] = rgb[2]
    }
}