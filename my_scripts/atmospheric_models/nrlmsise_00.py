from pyatmos import download_sw_nrlmsise00, read_sw_nrlmsise00
swfile = download_sw_nrlmsise00()
swdata = read_sw_nrlmsise00(swfile)