---
title: Homepage
date: '2019-08-07'
disable_mathjax: true
disable_highlight: true
---

### About Me
<img class="profile-picture" src="/images/Zhiqiang.png">
I'm Zhiqiang Wang, I'm working on computer vision and image processing, espacially on object detection and image captioning.

I did my Master’s degree in computational mathematics at the [Capital Normal University](http://eng.cnu.edu.cn).

### Projects

- sightseq (Image captioning, work in process)
  <a href="https://github.com/zhiqwang/sightseq" style="border-bottom:none;padding-bottom:0px;">
    <img src="https://img.shields.io/github/stars/zhiqwang/sightseq.svg?color=teal&logo=github"alt="GitHub stars">
  </a>
  <a href="https://github.com/zhiqwang/sightseq/forks" style="border-bottom:none;padding-bottom:0px;">
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
