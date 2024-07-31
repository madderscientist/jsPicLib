/**
 * 矩阵乘法
 * @param {Array<Aarray>} a 矩阵
 * @param {Array<Aarray>} b 矩阵
 * @returns 矩阵 失败则返回null
 */
function mul(a, b) {
    let row = a.length;
    let col = b[0].length;
    let t = b.length;
    if (t != a[0].length) return null;
    let result = Array(row);
    for (let r = 0; r < row; r++) {
        let v = Array(col).fill(0);
        for (let c = 0; c < col; c++) {
            for (let i = 0; i < t; i++) {
                v[c] += b[i][c] * a[r][i];
            }
        } result[r] = v;
    } return result;
}
/**
 * 转置
 * @param {Array<Aarray>} A 矩阵
 * @returns A^T
 */
function transpose(A) {
    let row = A.length;
    let col = A[0].length;
    let result = Array(col);
    for (let c = 0; c < col; c++) {
        let v = Array(row);
        for (let r = 0; r < row; r++) {
            v[r] = A[r][c];
        }
        result[c] = v;
    }
    return result;
}

/**
 * 矩阵求逆
 * @param {Array<Aarray>} square 矩阵
 * @returns M的逆矩阵 失败则返回null
 */
function inv(square) {
    let detValue = det(square);
    if(detValue == null || detValue == 0) return null;
    let result = adjoint(square);   // 上一条已经保证不是null了

    for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < result.length; j++) {
            result[i][j] /= detValue;
        }
    } return result;
}

/**
 * 求伴随矩阵
 * @param {Array<Array>} square 
 * @returns 伴随矩阵 失败则返回null
 */
function adjoint(square) {
    let n = square.length;
    if(n!=square[0].length) return null;
    let result = new Array(n).fill(0).map(arr => new Array(n).fill(0));
    for (let row = 0; row < n; row++) {
        for (let column = 0; column < n; column++) {
            // 去掉第 row 行第 column 列的矩阵
            let matrix = [];
            for (let i = 0; i < square.length; i++) {
                if (i !== row) {
                    let arr = [];
                    for (let j = 0; j < square.length; j++) {
                        if (j !== column) {
                            arr.push(square[i][j]);
                        }
                    } matrix.push(arr);
                }
            } result[row][column] = Math.pow(-1, row + column) * det(matrix);
        }
    } return transpose(result);
}

/**
 * 求行列式
 * @param {Array<Array>} square 
 * @returns 行列式 失败则返回null
 */
function det(square) {
    let n = square.length;
    if(n != square[0].length) return null;
    let result = 0;
    if (n > 3) {
        for (let column = 0; column < n; column++) {
            // 去掉第 0 行第 column 列的矩阵
            let matrix = new Array(n - 1).fill(0).map(arr => new Array(n - 1).fill(0));
            for (let i = 0; i < n - 1; i++) {
                for (let j = 0; j < n - 1; j++) {
                    if (j < column) {
                        matrix[i][j] = square[i + 1][j];
                    } else {
                        matrix[i][j] = square[i + 1][j + 1];
                    }
                }
            } result += square[0][column] * Math.pow(-1, 0 + column) * det(matrix);
        }
    } else if (n === 3) {
        // 3 阶
        result = square[0][0] * square[1][1] * square[2][2] +
            square[0][1] * square[1][2] * square[2][0] +
            square[0][2] * square[1][0] * square[2][1] -
            square[0][2] * square[1][1] * square[2][0] -
            square[0][1] * square[1][0] * square[2][2] -
            square[0][0] * square[1][2] * square[2][1];
    } else if (n === 2) 
        result = square[0][0] * square[1][1] - square[0][1] * square[1][0];
    else if (n === 1)
        result = square[0][0];
    return result;
}

/**
 * 逐项加法
 * @param {Array<Array>} a 
 * @param {Array<Array>} b 
 * @param {Array<Array>} to 如果不传就是加到a上面
 * @returns {Array<Array>} 失败则返回null
 */
function Add(a, b, to) {
    if(!to) to = a;
    let h = a.length;
    let w = a[0].length;
    if(h!=b.length || w!=b[0].length) return null;
    for(let i = 0; i < h; i++) {
        for(let j = 0; j < w; j++) {
            to[i][j] = a[i][j] + b[i][j];
        }
    } return to;
}