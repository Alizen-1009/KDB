# CUDA 多流如何用 event 重叠拷贝与计算？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 4、11 题

## 原题与条件

4. 如何利用 CUDA 流（Stream）实现多流并发？请说明如何避免流之间的虚假依赖。
11. 解释如何通过流水线技术（pipeline overlap）实现计算与数据拷贝的重叠，并给出使用 CUDA streams 和 events 的典型代码片段。

## 30 秒回答

同一 stream 有序。不同 stream 在依赖和硬件允许时并发。用 non-blocking stream，真正的跨流依赖用 record/wait event；避免热路径设备级同步。

## 深入解释

三个 non-blocking stream，host buffers 已 pin，两个 slot 各有独立 device buffer 及预创建的 ready/computed/done events。

```cpp
for (int i = 0; i < chunks; ++i) {
    int s = i % 2;
    if (i >= 2) cudaEventSynchronize(done[s]);
    cudaMemcpyAsync(d_in[s], h_in[i], in_bytes[i], cudaMemcpyHostToDevice, h2d);
    cudaEventRecord(ready[s], h2d);
    cudaStreamWaitEvent(compute, ready[s], 0);
    kernel<<<grid[i], block, 0, compute>>>(d_in[s], d_out[s], n[i]);
    cudaEventRecord(computed[s], compute);
    cudaStreamWaitEvent(d2h, computed[s], 0);
    cudaMemcpyAsync(h_out[i], d_out[s], out_bytes[i], cudaMemcpyDeviceToHost, d2h);
    cudaEventRecord(done[s], d2h);
}
cudaStreamSynchronize(d2h);
```

示例省略初始化、错误检查和释放。等 D2H 完成后才能复用该 slot，硬件 copy engines 和资源竞争决定实际重叠。GDN prefill/decode 可以用独立 scratch、同请求 recurrent state 的真实依赖，以及消费者完成后才复用 buffer 来说明。用 Nsight Systems 验证 overlap。

截图系列追问：增加 stream 数只是增加可调度工作，copy engine、SM 资源或内部工作队列饱和时可能只是排队、吞吐不再提升，不必然报错。检查真实依赖、buffer 生命周期及错误码，以时间线和端到端时间选择并发度，不承诺一定三向重叠。

## 关联知识

- [[../../../../wiki/concepts/通信-计算重叠|通信-计算重叠]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]（用户提供的截图系列摘要，第 11 题）
- [[../README|秋招问题汇总]]
