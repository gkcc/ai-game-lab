import io
import hashlib
import unittest
import zipfile
import numpy as np
from . import numeric


def arrays():
    return dict(shared=np.zeros((1,275),np.float32),candidates=np.zeros((2,98),np.float32),
        offsets=np.array([0,2],np.int64),teacher_choice_label=np.array([1],np.int64),
        ops=np.array([0,1],np.uint8),seed=np.array([7],np.uint32))


class NumericTest(unittest.TestCase):
    def read(self,stream,*limits):
        return numeric.read(stream,*limits,hashlib.sha256(stream.getvalue()).hexdigest())

    def test_roundtrip_and_limits(self):
        stream=io.BytesIO()
        numeric.write_stream(stream,arrays(),1,2)
        loaded=self.read(stream,1,2,100000,100000)
        self.assertEqual(loaded['teacher_choice_label'].tolist(),[1])
        self.assertNotIn('status',loaded)
        for limits in ((0,2,100000,100000),(1,1,100000,100000),(1,2,1,100000),(1,2,100000,1)):
            with self.assertRaises(ValueError): self.read(stream,*limits)

    def test_invalid_arrays(self):
        for key,value in (('offsets',np.array([0,0],np.int64)),('teacher_choice_label',np.array([2],np.int64)),
                          ('ops',np.array([0,255],np.uint8)),('seed',np.array([7],np.int64))):
            data=arrays();data[key]=value
            with self.assertRaises(ValueError): numeric.validate(data,1,2)
        data=arrays();data['shared'][0,0]=np.nan
        with self.assertRaises(ValueError): numeric.validate(data,1,2)
        data=arrays();data['status']=np.array([0],np.uint8)
        with self.assertRaises(ValueError): numeric.validate(data,1,2)
        empty=dict(shared=np.empty((0,275),np.float32),candidates=np.empty((0,98),np.float32),
            offsets=np.array([0],np.int64),teacher_choice_label=np.empty(0,np.int64),ops=np.empty(0,np.uint8),seed=np.empty(0,np.uint32))
        with self.assertRaises(ValueError): numeric.validate(empty,1,2)

    def test_bad_header_and_extra_member(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            for key in numeric.KEYS:
                archive.writestr(key+'.npy',b'\x93NUMPY\x02\x00\xff\xff\xff\x7f')
        with self.assertRaises(ValueError): self.read(stream,1,2,100000,100000)
        stream=io.BytesIO(); numeric.write_stream(stream,arrays(),1,2)
        with zipfile.ZipFile(stream,'a') as archive: archive.writestr('status.npy',b'bad')
        with self.assertRaises(ValueError): self.read(stream,1,2,100000,100000)

    def test_mutation_after_snapshot_uses_original_bytes(self):
        source=io.BytesIO(); numeric.write_stream(source,arrays(),1,2)
        original=source.getvalue(); digest=hashlib.sha256(original).hexdigest()
        class Changing(io.BytesIO):
            def read(self,n=-1):
                data=super().read(n)
                self.seek(0); self.truncate(); self.write(b'changed')
                return data
        loaded=numeric.read(Changing(original),1,2,100000,100000,digest)
        self.assertEqual(loaded['teacher_choice_label'].tolist(),[1])
        with self.assertRaises(ValueError): numeric.read(io.BytesIO(original),1,2,100000,100000,'0'*64)


if __name__ == '__main__': unittest.main()
