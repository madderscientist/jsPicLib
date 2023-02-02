// cluster algorithm
// useful when houghlines
function mean_shift(data, R, critical = 0.01) {
    // 预处理: 距离变为距离平方, 复制原数组
    R = R * R;
    if (!isNaN(data[0])) data = Array.from(data, x => [x]);
    else data = Array.from(data, x => [...x]);
    let output = [];
    function dis2(a, b) { // 距离的平方
        let summ = 0;
        for (let i = 0; i < a.length; i++) summ += (a[i] - b[i]) ** 2;
        return summ;
    }
    while(data.length!=0) {
        let center = data[0];           // 随便选个点
        for (let r = 0; r < 30; r++) {  // 上限30次迭代
            let sum = Array.from(center, _ => 0);
            let classmate = [];         // 这个类的所有成员(记下标)
            for (let i = 0; i < data.length; i++) {
                if (dis2(center, data[i]) <= R) {    // 在范围内
                    for (let j = 0; j < sum.length; j++) sum[j] += data[i][j];
                    classmate.push(i);
                }
            }
            for (let j = 0; j < sum.length; j++) sum[j] = sum[j] / classmate.length;    // 计算新的中心
            if (dis2(sum, center) <= critical) { // 中心不再移动 成功分类
                data = data.filter((_, i) => !classmate.includes(i));  // 从data中删除点
                center = sum;
                break;
            }
            center = sum;
        }
        output.push(center);
    }
    return output;
}