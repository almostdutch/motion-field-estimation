# motion-field-estimation
Motion field estimation based on the gradient constraint equation with iterative refinement

### demo_motion_field_estimation.py<br/>
This demo shows how to estimate motion field between two 2D (single channel) same size images

**Dummy dataset**<br/>
Given rotation angle: -1.16<br/>
Given translation Row: 4.01 Col: 4.29<br/>

<p align="center">
  <img src="test_results/dummy_images.png" width="620" height="240"/>
</p>

**After 1 iteration**<br/>
Estimated motion field<br/>
<p align="center">
  <img src="test_results/test_result_motion_field_1iter.png" width="620" height="240"/>
</p>

Motion correction<br/>
<p align="center">
  <img src="test_results/test_result_corr_1iter.png" width="620" height="240"/>
</p>

**After 2 iteration**<br/>
Estimated motion field<br/>
<p align="center">
  <img src="test_results/test_result_motion_field_2iter.png" width="620" height="240"/>
</p>

Motion correction<br/>
<p align="center">
  <img src="test_results/test_result_corr_2iter.png" width="620" height="240"/>
</p>

**After 10 iteration**<br/>
Estimated motion field<br/>
<p align="center">
  <img src="test_results/test_result_motion_field_10iter.png" width="620" height="240"/>
</p>

Motion correction<br/>
<p align="center">
  <img src="test_results/test_result_corr_10iter.png" width="620" height="240"/>
</p>
