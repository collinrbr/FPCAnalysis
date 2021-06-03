#!/usr/bin/env python
import sys
import netCDF4

try:
    filename = sys.argv[1]
    startindex = int(sys.argv[2])
    endindex = int(sys.argv[3])
    val = int(sys.argv[4])

except:
    print("This script changes metadata values. Range is inclusive. Indexing starts at 0.")
    print("usage: " + sys.argv[0] + " filename startindex endindex val")
    sys.exit()

# #load file into python script
# ncin = Dataset(filename, 'rw', format='NETCDF4')
#
# metadata_out = ncin.variables['metadata'][:]
# print(metadata_out)
#
# for i in range(0,len(metadata_out)):
#     if(i >= startindex and i <= endindex):
#         metadata_out[i] = val
#
# ncout = Dataset(filename+'out', 'w', format='NETCDF4')
# ncout = ncin
#
# metadata = ncout.createVariable('metadata','f4',('x',))
# metadata.description = '1 = signature, 0 = no signature'
# metadata[:] = metadata_out[:]
#
# #save file
# ncout.close()

#from https://stackoverflow.com/questions/15141563/python-netcdf-making-a-copy-of-all-variables-and-attributes-but-one
toexclude = ['metadata']

with netCDF4.Dataset(filename) as src, netCDF4.Dataset(filename+'.editedmetadata', "w") as dst:
    # copy global attributes all at once via dictionary
    dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))
    # copy all file data except for the excluded
    for name, variable in src.variables.items():
        if name not in toexclude:
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name][:] = src[name][:]
            # copy variable attributes all at once via dictionary
            dst[name].setncatts(src[name].__dict__)

    #edit meta data
    metadata_in = src.variables['metadata'][:]
    for i in range(0,len(metadata_in)):
        if(i >= startindex and i <= endindex):
            metadata_in[i] = val
    metadata = dst.createVariable('metadata','f4',('x',))
    metadata.description = '1 = signature, 0 = no signature'
    metadata[:] = metadata_in[:]

    #fix typo (TODO: fix in source and remove)
    print(dst.__dict__)
    dst.ShockNormalAngle = dst.ShockNormalAngel
    del dst.ShockNormalAngel
    print(dst.__dict__)
