/**
 * @description extension of jsPic in the fileds of matrix transform
 * @author madderscientist.github
 * @DataStruct
 * matrix used here: [
 *      [x, x, x],
 *      [x, x, x],
 *      [x, x, x]
 * ] must be 3*3
 * @example
 * // After import this file, jsPic will be replaced by jsPicMat
 * // Firstly, register your matrix operate functions:
 * jsPic.mat.mul = (a, b) => matrix(a) * matrix(b);
 * jsPic.mat.inv = (a) => 1 / matrix(a);
 * let pic = jspic.transform([
 *      [0,1,0],
 *      [1,0,0],
 *      [0,0,1]
 * ]);
 */
class jsPicMat extends jsPic {
    static mat = {
        mul: typeof mul === 'undefined' ? null : mul,
        inv: typeof mul === 'undefined' ? null : inv,
        _trans: [[1, 0, 0], [0, -1, 0], [0, 0, 1]],     // 变换算子
        /**
         * Rotation matrix of right multiplication (transpose of common rotation matrices)
         * @param {number} rad 
         * @returns {Array<Array>} Rotation matrix
         */
        rotate: (rad) => {
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            return [
                [cos, sin, 0],
                [-sin, cos, 0],
                [0, 0, 1]
            ];
        }
    }

    /**
     * 对图片使用矩阵变换 矩阵是右乘的 坐标系和画布坐标系相同，因此需要注意：
     * 变换矩阵的第一行是变换后的x基底，第二行是变换后的y基底；
     * 取第二列的相反数和一般直角坐标系一样
     * @param {Array<Array>} T 变换矩阵，3*3则透视变换，2*2则平面变换
     * @returns {jsPicMat} 变换后的图片 如果失败则返回null
     */
    transform(T) {
        if (T.length == 2) {
            T = [
                [...T[0], 0],
                [...T[0], 0],
                [0, 0, 1]
            ];
        }
        const Tinv = jsPicMat.mat.inv(T);
        if (Tinv == null) return null;
        // 确定宽度和高度
        // 由于是线性变换，因此只要知道四个顶点的位置
        const p1 = [...T[2]];    // jsPicMat.mat.mul([[0,0,1]], T)[0];
        const p2 = jsPicMat.mat.mul([[this.width, 0, 1]], T)[0];
        const p3 = jsPicMat.mat.mul([[0, this.height, 1]], T)[0];
        const p4 = jsPicMat.mat.mul([[this.width, this.height, 1]], T)[0];
        p1[0] /= p1[2]; p1[1] /= p1[2];
        p2[0] /= p2[2]; p2[1] /= p2[2];
        p3[0] /= p3[2]; p3[1] /= p3[2];
        p4[0] /= p4[2]; p4[1] /= p4[2];
        const minX = Math.min(p1[0], p2[0], p3[0], p4[0]);
        const minY = Math.min(p1[1], p2[1], p3[1], p4[1]);
        const maxX = Math.max(p1[0], p2[0], p3[0], p4[0]);
        const maxY = Math.max(p1[1], p2[1], p3[1], p4[1]);
        const width = Math.ceil(maxX - minX);
        const height = Math.ceil(maxY - minY);
        const size = width * height;
        // 提前计算映射关系 为了提高缓存命中率，第一层循环是通道，因此预计算映射关系
        // 映射关系用线性组合来算，避免矩阵乘法
        const base = jsPicMat.mat.mul([[minX, minY, 1]], Tinv)[0];
        const Xoffset = Tinv[0];    // jsPicMat.mat.mul([[1, 0, 0]], Tinv)[0];
        const Yoffset = Tinv[1];    // jsPicMat.mat.mul([[0, 1, 0]], Tinv)[0];
        const xmap = new Float32Array(size);
        const ymap = new Float32Array(size);
        for (let y = 0, point = 0; y < height; y++) {
            const tempBase = [...base];
            for (let x = 0; x < width; x++) {
                xmap[point] = tempBase[0] / tempBase[2];
                ymap[point] = tempBase[1] / tempBase[2];
                tempBase[0] += Xoffset[0];
                tempBase[1] += Xoffset[1];
                tempBase[2] += Xoffset[2];
                point++;
            }
            base[0] += Yoffset[0];
            base[1] += Yoffset[1];
            base[2] += Yoffset[2];
        }
        const result = new jsPicMat().new(this.channel.length, width, height, [0, 0, 0, 0]);
        return result.bilinearMap(this, xmap, ymap);
    }

    /**
     * the transform T is applied in a coordinate system whose origin is in the center of the image, and axes are the same as common coordinate
     * @param {Array<Array>} T Transform matrix, 2*2 or 3*3
     * @returns {jsPicMat} If the transformed image fails, return null
     */
    commonCoordTrans(T) {
        if (T.length == 2) {
            T = [
                [...T[0], 0],
                [...T[0], 0],
                [0, 0, 1]
            ];
        }
        const o = [
            [1, 0, 0],
            [0, -1, 0],
            [-this.width >> 1, this.height >> 1, 1]
        ]
        T = jsPicMat.mat.mul(o, T);
        // 求逆
        o[2][0] = -o[2][0];
        o[2][1] = -o[2][1];
        return this.transform(jsPicMat.mat.mul(T, o));
    }

    /**
     * rotate by degrees
     * @param {number} deg 
     * @returns {jsPicMat}
     */
    rotate(deg) {
        // 坐标系变换相当于反方向旋转
        return this.transform(jsPicMat.mat.rotate(-deg / 180 * Math.PI));
    }

    /**
     * like opencv.GetPerspectiveTransform
     * derivation process: https://github.com/madderscientist/JianPuTrans/blob/main/README.md
     * @param {*} from 4*2 four origin vertices
     * @param {*} to 4*2 four vertices after transform
     * @returns {Array<Array>} Perspective Transform Matrix
     */
    static GetPerspectiveTransform(from, to) {
        let A = [
            [from[0][0], 0, from[1][0], 0, from[2][0], 0, from[3][0], 0],
            [from[0][1], 0, from[1][1], 0, from[2][1], 0, from[3][1], 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, from[0][0], 0, from[1][0], 0, from[2][0], 0, from[3][0]],
            [0, from[0][1], 0, from[1][1], 0, from[2][1], 0, from[3][1]],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [-from[0][0] * to[0][0], -from[0][0] * to[0][1], -from[1][0] * to[1][0], -from[1][0] * to[1][1], -from[2][0] * to[2][0], -from[2][0] * to[2][1], -from[3][0] * to[3][0], -from[3][0] * to[3][1]],
            [-from[0][1] * to[0][0], -from[0][1] * to[0][1], -from[1][1] * to[1][0], -from[1][1] * to[1][1], -from[2][1] * to[2][0], -from[2][1] * to[2][1], -from[3][1] * to[3][0], -from[3][1] * to[3][1]]
        ];
        let x = [to.flat()];
        let [[a11, a21, a31, a12, a22, a32, a13, a23]] = jsPicMat.mat.mul(x, jsPicMat.mat.inv(A));
        return [
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, 1]
        ];
    }
}

jsPic = jsPicMat;