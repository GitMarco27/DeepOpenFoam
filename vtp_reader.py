from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtk.util import numpy_support as VN

# from vtk import vtkUnstructuredGridWriter


reader = vtkXMLPolyDataReader()
reader.SetFileName('case/postProcessing/wallPressure/1000/airfoil.vtp')
reader.Update()

n_arrays = reader.GetNumberOfPointArrays()
for i in range(n_arrays):
    print(reader.GetPointArrayName(i))

n_points = reader.GetNumberOfPoints()
n_cells = reader.GetNumberOfCells()

p = VN.vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('p'))
print(p.shape)
coordinates = VN.vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
print(coordinates.shape)

triangles = VN.vtk_to_numpy(reader.GetOutput().GetCells().GetData())

# writer = vtkUnstructuredGridWriter()
# writer.SetInputData(reader.GetOutput())
# writer.SetFileName("Output.vtk")
# writer.Write()
