---
title: Homepage
date: '2019-08-07'
disable_mathjax: true
disable_highlight: true
---

### About Me
<img class="profile-picture" src="/images/Zhiqiang.png">

My name is Zhiqiang Wang. I'm doing open-source work on [sightseq](https://github.com/zhiqwang/sightseq). My current focus is on deep learning and image processing. Now I am looking for a job opportunities in industry (Beijing as the first option). Please feel free to [contact me](#contact-me).

Previously, I did my Master’s degree in computational mathematics at [Capital Normal University](http://eng.cnu.edu.cn) (Beijing, China) in 2017, where my research revolved around optimization, specifically compressed sensing and its application on computed tomography (CT).

Prior to that, I did my B.Sc. degree in 2013 from [Ningbo University](https://www.nbu.edu.cn/en/) (Zhejiang, China). At that time, I wanted to be a mathematician :-) obviously, now I'm a little far away from this dreaming.

### Contact Me
Best way to contact me would be via [email](mailto:zhiqwang@outlook.com). You can also find me on [GitHub](https://github.com/zhiqwang) (where I’m active almost every day) or [Twitter](https://twitter.com/zhiq_w) (I read Twitter about three times a week).

### Projects

- [sightseq](https://github.com/zhiqwang/sightseq): PyTorch implementation of text recognition and object detection (work in process), my current goal is to achieve the implementation of image captioning, it can also be viewed as the computer vision tools for [fairseq](https://github.com/pytorch/fairseq), my ultimate goal is to build a general and modular framework for vision and language multimodal research.
<a href="https://github.com/zhiqwang/sightseq" style="border-bottom:0px;padding-bottom:0px;">
  <img src="https://img.shields.io/github/stars/zhiqwang/sightseq.svg?color=teal&logo=github"alt="GitHub stars">
</a>
<a href="https://github.com/zhiqwang/sightseq/forks" style="border-bottom:0px;padding-bottom:0px;">
  <img src="https://img.shields.io/github/forks/zhiqwang/sightseq.svg" alt="GitHub forks">
</a>

### Publications
<div class="publications">
  <table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;"><tbody>
    <tr onmouseout="edgepreserved_stop()" onmouseover="edgepreserved_start()">
      <td style="padding:10px;width:25%;vertical-align:middle">
        <div class="one">
          <div class="two" id='edgepreserved_image'>
            <img src="/images/edgepreserved_after.png">
          </div>
          <img src="/images/edgepreserved_before.png">
        </div>
        <script type="text/javascript">
          function edgepreserved_start() {
            document.getElementById('edgepreserved_image').style.opacity = "1";
          }
          function edgepreserved_stop() {
            document.getElementById('edgepreserved_image').style.opacity = "0";
          }
          edgepreserved_stop()
        </script>
      </td>
      <td style="padding:10px;width:75%;font-size:13px;vertical-align:middle">
        <b>Image reconstruction method for the exterior problem with 1D edge-preserved diffusion and smoothing</b>
        <br>
        Jinqiu Xu, <b>Zhiqiang Wang</b>, Yunsong Zhao, Peng Zhang
        <br>
        <em>Proceedings of the 5th International Conference on Image Formation in X-ray Computed Tomography (CT-Meeting)</em> 2018
        <br><a href="/data/XuCTMeeting2018.bib">Bibtex</a>
        <br>The blurred edges are restored gradually by edge-preserving diffusion and edge-preserving smoothing.
      </td>
    </tr>
  </tbody></table>
</div>