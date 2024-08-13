/**
 * @description a picture tool
 * @author madderscientist.github
 * @DataStruct
 *  jsPic has three attributes:
 *  channel [
 *      channel_1 [
 *          Uint8ClampedArray_1 [value_1, value_2, ...],
 *          Uint8ClampedArray_2 [...],
 *          ...
 *      ],
 *      channel_2 [...],
 *      ...
 *  ]
 *  height
 *  width
 * @example
 *  let j = new jsPic().fromImageData(imagedata,'L')
 *  j = j.convolution({kernel:jsPic.Laplacian, picfun:(x=>Math.abs(x)>200?255:0), padding:[1,1]})
 */
class jsPic {
    // useful kernels
    static Gaussian = [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]];
    static Prewitt_H = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
    static Prewitt_V = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]];
    static Sobel_H = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    static Sobel_V = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    static Laplacian = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];
    /**
     * normalized Gaussian kernel
     * @param {number} w width or size
     * @param {number} h height (if h is not given, h will be set as w)
     * @returns {Array<Array>} 2D square array
     */
    static GaussianKernel(w, h = -1) {
        if(h <= 0) h = w;
        let kernel = new Array(h);
        let sum = 0;
        const cw = w >> 1;
        const ch = h >> 1;
        for (let i = 0; i < h; i++) {
            kernel[i] = new Array(w);
            for (let j = 0; j < w; j++) {
                kernel[i][j] = Math.exp(-((i - ch) ** 2 + (j - cw) ** 2) / 2);
                sum += kernel[i][j];
            }
        }
        for (let i = 0; i < h; i++)
            for (let j = 0; j < w; j++)
                kernel[i][j] /= sum;
        return kernel;
    }

    /**
     * new a jsPic from param
     * @param {number | Array} channel if number, initialize data filled with param 'fill'; if Array, use it directly
     * @param {number} width the picture's width. If channel is Array, this value will be automatically set
     * @param {number} height the picture's height
     * @param {Int8Array} fill fill[i] = channel{i}'s default value (only used when 'channel' is a number)
     * @returns {jsPic} overwirte and return itself
     */
    new(channel, width, height, fill = [255, 255, 255, 255]) {
        if (Array.isArray(channel)) {
            this.channel = channel;
            this.width = channel[0][0].length;
            this.height = channel[0].length;
        } else {
            this.width = width;
            this.height = height;
            this.channel = Array.from({ length: channel }, (_, i) => this.newChannel(fill[i]));
        }
        return this;
    }

    /**
     * new a channel filled with 'fill'
     * @param {number} fill default value
     * @returns {Array} a channel
     */
    newChannel(fill = 0) {
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
    cloneChannel(channel) { return Array.from(this.channel[channel], (line) => new Uint8ClampedArray(line)); }

    /**
     * deeply clone a jsPic
     * @returns {jsPic} copy
     */
    clone() { return new jsPic().new(Array.from(this.channel, (_, index) => this.cloneChannel(index))); }

    /**
     * get a pixel at (x,y)
     * @param {number} x
     * @param {number} y
     * @returns {Array} [channel_1, channel_2, ... ]
     */
    getPixel(x, y) {
        x = x | 0;
        y = y | 0;
        let p = new Uint8ClampedArray(this.channel.length);
        for (let i = 0; i < this.channel.length; i++) p[i] = this.channel[i][y][x];
        return p;
    }

    /**
     * traverse this.channel[channel], replacing each value with mapfn(value)
     * @param {number | Array} channel target channel index or channel object
     * @param {Function} mapfn input: each value; its output will be used
     */
    throughChannel(channel, mapfn = (v, x, y) => v) {
        if(typeof channel === 'number') channel = this.channel[channel];
        for (let h = 0; h < this.height; h++)
            for (let w = 0; w < this.width; w++)
                channel[h][w] = mapfn(channel[h][w], w, h);
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
     * @param {String} mode convert mode:'RGB'|'RGBA'|'L'|'1' (convert here is quicker than convert afterwards)
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
     * @param {String} mode 'L' | '1' | 'RGB'
     * @param {number} threshold only used when mode='1'; for otsu method please use .convert_1 after converting to 'L'
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
                return new jsPic().new([C]);
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
                    return new jsPic().new([C]);
                } else break;
            case 'RGB':
                if (this.channel.length == 3) return this.clone();
                else if (this.channel.length == 4)
                    return new jsPic().new(Array.from({ length: this.channel.length - 1 }, (_, index) => this.cloneChannel(index)));
                else break;
            default: console.error("unknown mode!"); return null;
        }
        console.error("channel number error!"); return null;
    }

    /**
     * Binarization one channel (change itself)
     * @param {number} channel target channel index
     * @param {number} threshold if -1, use otsu method
     */
    convert_1(channel, threshold = 127) {
        if(threshold<0) threshold = this.otsu(channel);
        console.log(threshold);
        this.throughChannel(channel, (x) => { return x > threshold ? 255 : 0; });
    }

    /**
     * statistic of the channel
     * @param {number} channel which channel to statistic
     * @returns {Uint32Array} histogram of the channel
     */
    histogram(channel = 0) {
        let hist = new Uint32Array(256);    // 用Uint16会溢出
        let c = this.channel[channel];
        for (let h = 0; h < this.height; h++) {
            let cc = c[h];
            for (let w = 0; w < this.width; w++)
                hist[cc[w]]++;
        } return hist;
    }

    /**
     * ostu method to get the binary threshold of a channel
     * @param {number} channel channel to get the threshold
     * @returns {number} threshold
     */
    otsu(channel = 0) {
        let hist = this.histogram(channel);
        let PointNum = this.width * this.height; // 求和
        let All = 0;
        for (let i = 0; i < 256; i++) All += i * hist[i];
        let bSum = 0;
        let bNum = 0;
        let wNum = 0;
        let varMax = 0;
        let threshold = 0;
        for (let i = 0; i < 256; i++) {
            bNum += hist[i];  // 左侧的像素数目
            if (bNum == 0) continue;
            wNum = PointNum - bNum;  // 右侧的像素数目
            if (wNum == 0) break;
            bSum += i * hist[i];    // 左侧的灰度总和
            let mean_dis = bSum / bNum - (All - bSum) / wNum;
            let varBetween = bNum * wNum * mean_dis * mean_dis;
            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = i;
            }
        } return threshold;
    }

    /**
     * an opening convolution focusing on each channel. All the operators is defind in pixfun, whose parameter is the 1D-Array of values masked by kernel
     * @param {Object} settings
     * @param {[number, number]} settings.kernel - kernel size: [width, height]. Odd number is suggested.
     * @param {[number, number]} settings.stride - [Xstride, Ystride]. Default is [1, 1]
     * @param {[number, number]} settings.padding - padding size: [paddingWidth, paddingHeight]. If -1, padding = kernelSize >> 1
     * @param {[number, number]} settings.fill - if fill[i] < 0, the nearliest pixel value will be used; else this.channel[select[i]] will be filled with fill[i].
     * @param {Function} settings.pixfun - how to process the kernel. Input is the 1D-Array of values masked by kernel, and its anchor coordinate, output is the pixel value.
     * @param {Array} settings.select - only picks channels to convolute, and its order doesn't matter. Unselected channels will be copied.
     * @returns new jspic, which has the same channel count as the original one
     */
    filter2D({ kernelSize = [3, 3], stride = [1, 1], padding = [-1, -1], fill = [127, 127, 127, 0], pixfun = (Ker, x, y) => Math.max(...Ker), select = [0, 1, 2] }) {
        let [kernelW, kernelH] = kernelSize;
        if(padding[0] < 0) padding[0] = kernelW >> 1;
        if(padding[1] < 0) padding[1] = kernelH >> 1;
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
                        let Ker = Array(kernelW * kernelH);
                        let ki = 0;
                        for (let hh = 0; hh < kernelH; hh++) {
                            for (let ww = 0; ww < kernelW; ww++, ki++) {
                                let kh = h + hh;
                                let kw = w + ww;
                                let value = fill[i];
                                let wjudge = kw < 0 || kw >= this.width;
                                let hjudge = kh < 0 || kh >= this.height
                                if (wjudge || hjudge) {
                                    if (value < 0) {    // 找最近的图片点
                                        if (wjudge) kw = Math.min(Math.max(0, kw), this.width - 1);
                                        if (hjudge) kh = Math.min(Math.max(0, kh), this.height - 1);
                                        value = this.channel[c][kh][kw];
                                    }
                                } else value = this.channel[c][kh][kw];
                                Ker[ki] = value;
                            }
                        } L[W] = pixfun(Ker, w, h);
                    } cha[H] = L;
                } C[c] = cha;
            } else {
                if (newHeight == this.height && newWidth == this.width) C[c] = this.cloneChannel(c);
                else {  // 若大小变化，多了填充255，少了剪裁，左上角对齐
                    console.warn('channel size not match!');
                    let Channel = new Array(newHeight);
                    for (let hh = 0, minh = Math.min(newHeight, this.height); hh < minh; hh++) {
                        let x = new Uint8ClampedArray(newWidth).fill(255);
                        for (let ww = 0, minw = Math.min(newWidth, this.width); ww < minw; ww++) {
                            x[ww] = this.channel[c][hh][ww];
                        } Channel[h] = x;
                    } C[c] = Channel;
                }
            }
        }
        return new jsPic().new(C);
    }

    /**
     * convoluion on selected channels
     * @param {Object} setting refer to filter2D
     * @param {Array} setting.kernel 2D array
     * @returns new jsPic
     */
    convolution({ kernel, stride = [1, 1], padding = [-1, -1], fill = [127, 127, 127, 0], pixfun = x => { return x; }, select = [0, 1, 2] }) {
        let kernelW = kernel[0].length;
        let kernelH = kernel.length;
        kernel = kernel.flat();
        return this.filter2D({
            kernelSize: [kernelW, kernelH], select: select,
            stride: stride, padding: padding, fill: fill,
            pixfun: Ker => {
                let sum = 0;
                for (let i = 0; i < kernel.length; i++) sum += kernel[i] * Ker[i];
                return pixfun(sum);
            }
        });
    }

    /**
     * similar to opencv, but the border uses neighbour, anchor is center
     * @param {Array} shapeKernel mask of shape, 0 means ignore
     * @returns a new jspic
     */
    erode(shapeKernel = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]) {
        let w = shapeKernel[0].length;
        let h = shapeKernel.length;
        let c = this.channel.length;
        shapeKernel = shapeKernel.flat();
        return this.filter2D({
            kernelSize: [w, h],
            padding: [Math.floor(w / 2), Math.floor(h / 2)],
            fill: Array(c).fill(-1),
            select: Array.from({ length: c }, (_, x) => x),
            pixfun: function (ker) {
                let min = 255;
                for (let i = 0; i < w * h; i++) {
                    if (shapeKernel[i])
                        if (ker[i] < min) min = ker[i];
                }
                return min;
            }
        });
    }

    /**
     * similar to opencv, but the border uses neighbour, anchor is center
     * @param {Array} shapeKernel 
     * @returns a new jspic
     */
    dilate(shapeKernel = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]) {
        let w = shapeKernel[0].length;
        let h = shapeKernel.length;
        let c = this.channel.length;
        shapeKernel = shapeKernel.flat();
        return this.filter2D({
            kernelSize: [w, h],
            padding: [Math.floor(w / 2), Math.floor(h / 2)],
            fill: Array(c).fill(-1),
            select: Array.from({ length: c }, (_, x) => x),
            pixfun: function (ker) {
                let max = 0;
                for (let i = 0; i < w * h; i++) {
                    if (shapeKernel[i])
                        if (ker[i] > max) max = ker[i];
                }
                return max;
            }
        });
    }

    /**
     * change selected channels' brightness
     * @param {String} mode 'gamma' | 'hsb' | 'linear'
     * @param {number} extent >0 meaning changes with mode
     * @param {Array} channels target channels' index
     * @returns {jsPic | null} change and return itself when succeeded. otherwise null
     */
    brighten(mode = 'gamma', extent = 1, channels = [0, 1, 2]) {
        switch (mode) {
            case 'gamma':
                extent = 1 / extent;
                let gammaMap = Array.from({ length: 256 }, (_, i) => 255 * Math.pow(i / 255, extent));
                for (let i = 0; i < channels.length; i++) this.throughChannel(i, x => gammaMap[x]);
                return this;
            case 'hsb':
                if (this.channel.length < 3 || channels.length != 3) break;
                return this.throughPic((pixel) => {
                    let hsb = rgbToHsb([pixel[channels[0]], pixel[channels[1]], pixel[channels[2]]]);
                    hsb[2] = Math.min(100, hsb[2] * extent);
                    let result = hsbToRgb(hsb);
                    for (let c = 0; c < 3; c++) pixel[channels[c]] = result[c];
                    return pixel;
                });
            case 'linear':
                for (let c = 0; c < channels.length; c++) {
                    this.throughChannel(channels[c], x => extent * x);
                    return this;
                }
            default: console.error("unknown mode!"); return null;
        }
        console.error("channel number error!"); return null;
    }

    /**
     * histogram equalization
     * @param {number} channel channel to apply
     * @returns this
     */
    equalizeHist(channel = 0) {
        let hist = new Array(256).fill(0);
        let c = this.channel[channel];
        // 求直方图
        for (let h = 0; h < this.height; h++)
            for (let w = 0; w < this.width; w++)
                hist[c[h][w]]++;
        let sum = hist.reduce((a, b) => a + b); // 求和
        let map = new Array(256);
        let sumHist = 0;
        for (let i = 0; i < 256; i++) {
            sumHist += hist[i];
            map[i] = Math.round(255 * sumHist / sum);   // 概率密度对应到像素
        }
        this.throughChannel(channel, x => map[x]);
        return this;
    }

    /**
     * fill all the holes satisfying the judge (255 is regarded as edge)
     * @param {number} channel target channel index
     * @param {*} judge given the area points array and perimeter points array, return ture means fill the area
     * @returns this
     */
    fillHole(channel = 0, judge = (area, perimeter) => true) {
        let copy = this.cloneChannel(channel);
        channel = this.channel[channel];
        let dx = [0, 0, 1, -1];
        let dy = [1, -1, 0, 0];
        let area, perimeter;
        let height = this.height, width = this.width;
        // 基于栈查找封闭区域
        function seek(seed = [0, 0]) {
            copy[seed[1]][seed[0]]=1;
            let stack = [seed];
            area = [];
            perimeter = [];
            let edge = true;
            while (stack.length) {
                let at = stack.pop();
                area.push(at);
                copy[at[1]][at[0]]=1;
                for (let i = 0; i < 4; i++) {
                    let nx = at[0] + dx[i];
                    let ny = at[1] + dy[i];
                    if (0 <= ny && ny < height && 0 <= nx && nx < width) {
                        if (copy[ny][nx] != 1) {
                            if (copy[ny][nx] == 255) perimeter.push([nx, ny]);
                            else {                      // 如果为0就入栈
                                stack.push([nx, ny]);
                                copy[ny][nx] = 1;       // 用1标记已访问
                            }
                        }
                    } else edge = false;    // 没有边界
                }
            } return edge;
        }
        for (let h = 0; h < this.height; h++) {
            for (let w = 0; w < this.width; w++) {
                if (copy[h][w] == 0) {
                    if (seek([w, h])) {
                        if (judge(area, perimeter)) {
                            for (let i = 0; i < area.length; i++) channel[area[i][1]][area[i][0]] = 255;
                        }
                    }
                }
            }
        }
        return this;
    }

    /**
     * resize via bilinear interpolation
     * @param {number} w if <=0, keeps the ratio
     * @param {number} h if <=0, keeps the ratio
     * @returns a new jspic (null if failed)
     */
    resize(w, h=-1) {
        if(w <= 0) {
            if(h > 0) w = this.width * h / this.height;
            else return null;
        }
        if(h <= 0) h = this.height * w / this.width;
        w = Math.round(w);
        h = Math.round(h);
        if(w <= 0 || h <= 0) return null;
        // 用双线性插值 映射法则: 放大的时候用点模型 缩小的时候用边模型
        let pw = w < this.width ? this.width / w : (this.width - 1.5) / (w - 1);
        let xmap = pw < 1 ? Array.from({ length: w }, (_, x) => x * pw) : Array.from({ length: w }, (_, x) => (x + 0.5) * pw - 0.5);
        let xl = new Float32Array(w);
        let xr = new Float32Array(w);
        for (let i = 0; i < w; i++) {
            let intx = Math.floor(xmap[i]);
            xl[i] = xmap[i] - intx;
            xr[i] = 1 - xl[i];
            xmap[i] = intx;
        }
        let ph = h < this.height ? this.height / h : (this.height - 1.5) / (h - 1);
        let ymap = ph < 1 ? Array.from({ length: h }, (_, y) => y * ph) : Array.from({ length: h }, (_, y) => (y + 0.5) * ph - 0.5);
        let yl = new Float32Array(h);
        let yr = new Float32Array(h);
        for (let i = 0; i < h; i++) {
            let inty = Math.floor(ymap[i]);
            yl[i] = ymap[i] - inty;
            yr[i] = 1 - yl[i];
            ymap[i] = inty;
        }
        let output = new jsPic().new(this.channel.length, w, h);
        for (let c = 0; c < this.channel.length; c++) {
            let ch = this.channel[c];
            for (ph = 0; ph < h; ph++) {
                for (pw = 0; pw < w; pw++) {
                    output.channel[c][ph][pw] =
                        xr[pw] * yr[ph] * ch[ymap[ph]][xmap[pw]] +          // 左上角
                        xl[pw] * yl[ph] * ch[ymap[ph] + 1][xmap[pw] + 1] +  // 右下角
                        xr[pw] * yl[ph] * ch[ymap[ph] + 1][xmap[pw]] +      // 左下角
                        xl[pw] * yr[ph] * ch[ymap[ph]][xmap[pw] + 1];       // 右上角
                }
            }
        }
        return output;
    }

    /**
     * Template Matching, using sum of squares of differences
     * @param {jsPic} template template picture
     * @param {number} ignorePix if template's pixel==ignorePix, it won't be calculated
     * @returns error map
     */
    TemplateMatch(template, ignorePix = -1) {
        let newW = this.width - template.width + 1;
        let newH = this.height - template.height + 1;
        let error = new Uint32Array(newH * newW);
        for (let c = 0; c < this.channel.length; c++) {
            let k = 0;
            for (let h = 0; h < newH; h++) {
                for (let w = 0; w < newW; w++, k++) {
                    // 遍历模版
                    for (let th = 0; th < template.height; th++) {
                        for (let tw = 0; tw < template.width; tw++) {
                            if (template.channel[c][th][tw] != ignorePix)
                                error[k] += (template.channel[c][th][tw] - this.channel[c][h + th][w + tw]) ** 2;
                        }
                    }
                }
            }
        }
        return error;
    }

    /**
     * copy part of jspic
     * @param {number} x left
     * @param {number} y top
     * @param {number} w area width
     * @param {number} h area height
     * @returns new jspic
     */
    copy(x,y,w,h){
        let p = new jsPic().new(this.channel.length,w,h,[0,0,0,0]);
        w = Math.min(x+w,this.width);
        h = Math.min(y+h,this.height);
        for(let yy=0;y<h;y++,yy++)
            for(let j=x, xx=0; j<w; j++, xx++)
                for(let c=0;c<this.channel.length;c++)
                    p.channel[c][yy][xx] = this.channel[c][y][j];
        return p;
    }

    /**
     * mix jspic with another
     * @param {number} x left
     * @param {number} y top
     * @param {jsPic} jspic to be pasted
     * @param {function} picfun how to mix. Inputs are two pixels, return an Array
     */
    paste(x, y, jspic, picfun = (origin,cover)=>cover){
        x = Math.round(x);
        y = Math.round(y);
        let h = Math.min(this.height, jspic.height + y);
        let w = Math.min(this.width, jspic.width + x);
        for(let yy=0; y<h; y++, yy++)
            for(let j=x, xx=0; j<w; j++, xx++){
                let pix = picfun(
                    Array.from(this.channel,(_,i)=>this.channel[i][y][j]),
                    Array.from(jspic.channel,(_,i)=>jspic.channel[i][yy][xx])
                );
                for(let c=0; c<this.channel.length; c++)
                    this.channel[c][y][j] = typeof pix[c] === 'number'? pix[c] : 0;
            }
    }

    /**
     * find the connected components: 8 neighborhood judgment connectivity
     * @param {number} channel channel to be processed
     * @param {number} threshold bigger than which is called "connected"
     * @returns [map, position]: map is the label map, position is the box of each component: [x,y,w,h,state(t or f)]
     */
    connectedComponents(channel = 0, threshold = 127) {
        channel = this.channel[channel];
        let map = new Array(this.height);   // 标记图
        for (let h = 0; h < this.height; h++)
            map[h] = new Uint16Array(this.width).fill(-1);
        let maxValue = map[0][0];
        let position = [];              // 位置表
        let id = 0;
        let seek = (x, y) => {
            let state = channel[y][x] > threshold;
            let locL = this.width, locR = 0, locT = this.height, locB = 0;  // 位置
            let stackX = [x], stackY = [y]; // 不用queue是因为shift开销大，用两个stack是因为比一个快了5倍
            while (stackX.length) {
                let w = stackX.pop();
                let h = stackY.pop();
                if (w < 0 || w >= this.width || h < 0 || h >= this.height) continue;
                if (map[h][w] != maxValue || state != (channel[h][w] > threshold)) continue;
                if (w < locL) locL = w;
                if (w > locR) locR = w;
                if (h < locT) locT = h;
                if (h > locB) locB = h;
                map[h][w] = id;
                stackX.push(w + 1, w - 1, w, w, w + 1, w - 1, w + 1, w - 1);
                stackY.push(h, h, h + 1, h - 1, h + 1, h + 1, h - 1, h - 1);
            }
            position.push([locL, locT, locR - locL + 1, locB - locT + 1, state]);
            id++;
        }
        for (let h = 0; h < this.height; h++)
            for (let w = 0; w < this.width; w++)
                if (map[h][w] == maxValue) seek(w, h);
        return [map, position];
    }

    /**
     * adaptiveThreshold
     * @param {number} channel which channel to process
     * @param {number | Array<Array>} KernelOrSize kernel or is width
     * @param {number} c the same meaning as OpenCV
     * @param {number} fill padding option like filter2D
     * @returns new jsPic with one channel
     */
    adaptiveThreshold(channel, KernelOrSize=jsPic.Gaussian, c=2, fill=-1) {
        let kernel = KernelOrSize;
        if (typeof KernelOrSize === 'number') {
            kernel = Array.from({ length: KernelOrSize }, () => Array(KernelOrSize).fill(1));
        }
        let kernelSum = 0;
        for (let i = 0; i < kernel.length; i++)
            for (let j = 0; j < kernel[i].length; j++)
                kernelSum += kernel[i][j];
        let kernelh = kernel.length;
        kernel = kernel.flat();
        channel = this.channel[channel];
        let filtered = new jsPic().new([channel]);   // 引用
        filtered = filtered.filter2D({
            kernelSize: [kernelh, kernelh],
            fill: [fill],
            pixfun: Ker => {
                let sum = 0;
                for (let i = 0; i < Ker.length; i++) sum += kernel[i] * Ker[i];
                return (sum / kernelSum);
            }
        });
        let fc = filtered.channel[0];
        for(let i = 0; i < this.height; i++)
            for(let j = 0; j < this.width; j++)
                fc[i][j] = channel[i][j] > fc[i][j] - c ? 255 : 0;
        return filtered;
    }

    /**
     * Hough Transform
     * @param {number} channel target channel index
     * @param {number} threshold voter critical
     * @param {number} rho r accuracy
     * @param {number} theta θ accuracy
     * @param {boolean} kb_line if return the k b of each line
     * @returns [[k array],[b array]] if kb_line else [[θ array],[r array]]
     */
    Hough(channel = 0, threshold = 95, rho = 1, theta = 1, kb_line = true) {
        rho = 1 / rho;
        channel = this.channel[channel];
        // 获取cos sin取值表 0~180
        let thetaL = parseInt(180 / theta); // 横轴长度
        theta = Math.PI / 180 * theta;
        let sinMap = new Float32Array(thetaL);
        let cosMap = new Float32Array(thetaL);
        if (theta == Math.PI / 180) {     // 速度上小小的优化
            for (let i = 0, angle = 0; i < 90; i++, angle += theta) {
                let value = Math.cos(angle);
                sinMap[i + 90] = sinMap[90 - i] = cosMap[i] = value * rho;// js特性: 越界赋值不报错也不成功
                cosMap[180 - i] = -value * rho;
            }
        } else {
            for (let i = 0, angle = 0; i < thetaL; i++, angle += theta) {
                sinMap[i] = Math.sin(angle) * rho;
                cosMap[i] = Math.cos(angle) * rho;
            }
        }
        // 投票数组
        let rmax = Math.round(Math.max(this.width, this.height) * rho);
        let Rrange = Math.round(Math.sqrt(Math.pow(this.width, 2) + Math.pow(this.height, 2)) * rho) + rmax + 1;
        let lineArea = new Uint16Array(Rrange * thetaL);
        // theta - x, r - y
        for (let h = 0; h < this.height; h++) {
            for (let w = 0; w < this.width; w++) {
                if (channel[h][w] != 255) continue;
                for (let theta = 0; theta < thetaL; theta++) {
                    let r = Math.round(w * cosMap[theta] + h * sinMap[theta]) + rmax;
                    lineArea[thetaL * r + theta]++;
                }
            }
        }
        let ok = new Array();
        for (let i = 0; i < lineArea.length; i++)
            if (lineArea[i] > threshold) ok.push(i);
        let k = new Array(ok.length);
        let b = new Array(ok.length);
        if (kb_line) {
            for (let i = 0; i < ok.length; i++) {
                let theta = ok[i] % thetaL;
                let r = parseInt(ok[i] / thetaL) - rmax;
                k[i] = -cosMap[theta] / (sinMap[theta] + 0.0001);
                b[i] = r / (sinMap[theta] + 0.0001);
            }
        } else {
            for (let i = 0; i < ok.length; i++) {
                k[i] = ok[i] % thetaL;
                b[i] = parseInt(ok[i] / thetaL) - rmax;
            }
        }
        return [k, b];
    }

    /**
     * Progressive Probabilistic Hough Transform
     * @param {number} channel target channel index
     * @param {number} threshold voter critical
     * @param {number} lineLength minimum line length
     * @param {number} lineGap how long is the line allowed to be disconnected
     * @param {number} rho r accuracy
     * @param {number} theta θ accuracy
     * @returns [[line1_start_point[x,y], line1_end_point], [line2...],...]
     */
    HoughP(channel = 0, threshold = 95, lineLength = 95, lineGap = 2, rho = 1, theta = 1) {
        rho = 1 / rho;
        let thetaL = parseInt(180 / theta);             // 横轴长度
        theta = Math.PI / 180 * theta;                  // 弧度精度
        let rmax = Math.round(Math.max(this.width, this.height) * rho);   // 负的最大距离
        let rL = Math.round(Math.sqrt(Math.pow(this.width, 2) + Math.pow(this.height, 2)) * rho + rmax + 1);   // 纵轴长度
        // 计算正余弦表
        let sinMap = new Float32Array(thetaL);
        let cosMap = new Float32Array(thetaL);
        if (theta == Math.PI / 180) {     // 速度上小小的优化
            for (let i = 0, angle = 0; i < 90; i++, angle += theta) {
                let value = Math.cos(angle);
                sinMap[i + 90] = sinMap[90 - i] = cosMap[i] = value * rho;
                cosMap[180 - i] = -value * rho;
            }
        } else {
            for (let i = 0, angle = 0; i < thetaL; i++, angle += theta) {
                sinMap[i] = Math.sin(angle) * rho;
                cosMap[i] = Math.cos(angle) * rho;
            }
        }
        let lineArea = new Uint16Array(rL * thetaL);    // 霍夫空间 投票器

        channel = this.channel[channel];
        let flag = new Array(this.height);              // 状态矩阵
        let edges = [];                                 // 列出所有边界点
        let output = [];
        for (let h = 0; h < this.height; h++) {
            let aLine = new Uint8Array(this.width);
            for (let w = 0; w < this.width; w++)
                if ((aLine[w] = channel[h][w]) == 255) edges.push([w, h]);//就是单等号
            flag[h] = aLine;
        }
        console.log(edges.length)
        for (let count = edges.length; count > 0;) {
            // Step1. 随机选点
            let i = Math.floor(Math.random() * count);
            let p = edges[i];               // [w,h]
            edges[i] = edges[--count];      // 删除这个点: 覆盖 
            if (!flag[p[1]][p[0]]) continue;// 被处理过(已经属于其他直线)\
            // Step2. 投票
            let max_vote = threshold - 1;
            let max_x = 0;
            for (let x = 0; x < thetaL; x++) {
                let r = Math.round(p[0] * cosMap[x] + p[1] * sinMap[x]) + rmax;
                let current_vote = ++lineArea[thetaL * r + x];
                if (current_vote > max_vote) {
                    max_vote = current_vote;
                    max_x = x;
                }
            }
            // Step3. 此点没有带来大于阈值的投票则下一个点
            if (max_vote < threshold) continue;
            // Step4. 从此点出发搜索端点
            // 为了能检测水平和竖直必须选变化小的步长为1
            let dx = 1, dy = 1;
            let SIN = sinMap[max_x];
            let COS = cosMap[max_x];
            let endPoint = [[0, 0], [0, 0]];
            if (SIN > Math.abs(COS))    // 在45~135°之间
                dy = -COS / SIN;        // r=xcos(θ)+ysin(θ) => y=-cos/sin*x+r/sin
            else                        // 直线较陡峭 应选取x步长为1
                dx = -SIN / COS;
            for (let k = 0; k < 2; k++, dx = -dx, dy = -dy) {
                let GAP = 0;
                for (let x0 = p[0], y0 = p[1]; ; y0 += dy, x0 += dx) {
                    let inty = Math.round(y0), intx = Math.round(x0);
                    if (inty < 0 || inty >= this.height || intx < 0 || intx >= this.width) break;
                    if (flag[inty][intx] == 255) {     // 该点未被访问 在线上
                        GAP = 0;
                        endPoint[k][0] = x0;
                        endPoint[k][1] = y0;
                    } else if (++GAP > lineGap) break;
                }
            }
            max_vote = Math.sqrt(Math.pow(endPoint[1][1] - endPoint[0][1], 2) + Math.pow(endPoint[1][0] - endPoint[0][0], 2)) >= lineLength;
            // 第二次循环 更新flag和投票器
            for (let k = 0; k < 2; k++, dx = -dx, dy = -dy) {
                for (let x0 = p[0], y0 = p[1]; x0 != endPoint[k][0] || y0 != endPoint[k][1]; y0 += dy, x0 += dx) {
                    let intx = Math.round(x0), inty = Math.round(y0);
                    if (flag[inty][intx] == 255) {
                        if (max_vote) {
                            for (let a = 0; a < thetaL; a++)
                                lineArea[thetaL * (Math.round(intx * cosMap[a] + inty * sinMap[a]) + rmax) + a]--;
                        }
                        flag[inty][intx] = 0;
                    }
                }
            }
            if (max_vote) output.push(endPoint);
        }
        return output;
    }

    /**
     * bilinear interpolation. map is an array of size width*height
     * @param {jsPic} source map source
     * @param {Float32Array} xmap 
     * @param {Float32Array} ymap 
     * @param {Array} fill if fill[i]<0, the closest pixel will be used 
     * @returns {jsPic} this
     */
    bilinearMap(source, xmap, ymap, fill = [-1, -1, -1, 0]) {
        const size = this.width*this.height;
        if(xmap.length!=size || ymap.length!=size) throw new Error("size not match");
        for (let c = 0; c < this.channel.length; c++) {
            const sch = source.channel[c];
            for (let y = 0, point = 0; y < this.height; y++) {
                const row = this.channel[c][y];
                for (let x = 0; x < this.width; x++) {
                    const X = xmap[point];
                    const Y = ymap[point];
                    point++;
                    // 如果坐标在原图像内
                    if (X >= -1 && X < source.width && Y >= -1 && Y < source.height) {
                        // 双线性插值
                        const x0 = Math.floor(X);   // 必须用floor，因为如果是-0.4，用 |0 得到的是0
                        const x1 = Math.ceil(X);
                        const y0 = Math.floor(Y);
                        const y1 = Math.ceil(Y);
                        const dx = X - x0; const bx = 1 - dx;
                        const dy = Y - y0; const by = 1 - dy;
                        // 如果越界则使用fill[c]
                        let row_must = sch[y0] || sch[y1];
                        let row_y0 = sch[y0] || [];
                        let row_y1 = sch[y1] || [];
                        const FILL = fill[c] < 0 ? ((row_must[x0] || row_must[x1]) | 0) : fill[c];
                        const x0y0 = row_y0[x0] == void 0 ? FILL : row_y0[x0];
                        const x1y0 = row_y0[x1] == void 0 ? FILL : row_y0[x1];
                        const x0y1 = row_y1[x0] == void 0 ? FILL : row_y1[x0];
                        const x1y1 = row_y1[x1] == void 0 ? FILL : row_y1[x1];
                        row[x] = bx * by * x0y0 + dx * by * x1y0 + bx * dy * x0y1 + dx * dy * x1y1;
                    }
                }
            }
        } return this;
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
    rgb.sort((a, b) => { return a - b; });
    let max = rgb[2];
    let min = rgb[0];
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


function convolution(ImgData, kernel, stride = 1, padding = 0, fill = [127, 127, 127, 0], pixfun = x => { return x; }) {
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