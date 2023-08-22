# Interactive_ML_Terrain

Introduction:

It is currently a pytorch version of WGAN model for automatically generating Height Map or digital elevation model(DEM) for terrain. In the future, it will be gradually updated to a CGAN model and accept user sketch inputs which is to indicate river and ridge. 

Dependancy:

1. h5py: This module allows to store large quantity of DEMs and allows very fast read and write.
   
2. pytorch: pytorch is very powerful module for machine learning coding. Compared to TensorFlow,       imcompatibility seldom occurs.

3. webdataset: In WGAN_GP implementation, data is first stored in .hdf file and loads into map-style dataset. The drawback of map-style dataset is to load all the data in one time which may cause memory error if you run with small RAM. However, webdataset allows stream data via urls, using less memory without losing speed if you have ssd. You can store your data on local computer, local server or online(flexible).
     1) Your dataset should in .tar format. Then for each training sample, if you will have two images, they should be named like "prefix.filename.format",          "prefix.filename1.format". All the files will be used in the same sample need the same prefix and "filename.format" will be used to distinguish different files.
     2) Followings are some web pages, helping you learn more about webdataset. \
        webdataset official webpage: https://webdataset.github.io/webdataset/ \
        command for creating directory: https://www.gnu.org/software/tar/manual/html_node/directory.html \
        online examples: https://medium.com/red-buffer/why-did-i-choose-webdataset-for-training-on-50tb-of-data-98a563a916bf \
        !!very important youtuber video source for webdataset: https://www.youtube.com/@t.m.breuel2670
