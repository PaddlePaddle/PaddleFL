## Data Alignment Tool

This is an example of using the `alignment` function to build a command line tool of PSI (Private Set Intersection).

### Usage

```bash
python align.py --party_id=$PARTY_ID --endpoints=$END_POINTS --data_file=$FILE_NAME [--is_receiver]
```  
### Example

Take data alignment between two parties , e.g., Alice (whose party_id is 0, IP address is 'A.A.A.A', port is 11111) and Bob (whose party_id is 1, IP address is 'B.B.B.B', port is 22222), as an example. Alice and Bob would like to find the intersection of alice_data.txt and bob_data.txt respectively, and Bob is intended to receive the final result. 

On each party:

*  **Alice**

```bash
python align.py --party_id=0 --endpoints=0:A.A.A.A:11111,1:B.B.B.B:22222 --data_file=alice_data.txt
```

*  **Bob**

```bash
python align.py --party_id=1 --endpoints=0:A.A.A.A:11111,1:B.B.B.B:22222 --data_file=bob_data.txt --is_receiver
