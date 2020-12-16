# Seam Carving
- In the file **_SC_bb.jl_** we have the module containing all the functions to perform seam carving.
- In the module **_Tst_bb.jl_** we have all the assertions check that we inserted to ensure that the efficient new version of functions gave the same results of the previous one. Also commented we have some lines of code to assess the performance of the code.
- To look at a detailed description of the module and how it works look at report **_Seam_Carving_report.pdf_**.

-To run the code, just clone the repository, and after having instantiated all the required modules and versions run the following code:
```
include("SB_bb.jl")
image = (load("./images/seam_carving.jpg"))
aux = SC_bb.initialize_aux(image')
new_img=SC_bb.seam_carve(aux,iterations);
```
Where iterations is the desired number of iterations to perform on the test image we chose.
To run the adaptive version of the code, run:
```
image = (load("./images/seam_carving.jpg"))
aux=SC_bb.initialize_aux(image)
new_img=SC_bb.ada_seam_carve(aux,iterations)
```
Note that in this case, since we want the algorithm to decide if to remove rows or columns, we do not tranpose `image` when initializing the auxiliary mutable struct.

- In the notebook there are some examples with printed output of how the module works, and also the codes that generated the images of the report.

