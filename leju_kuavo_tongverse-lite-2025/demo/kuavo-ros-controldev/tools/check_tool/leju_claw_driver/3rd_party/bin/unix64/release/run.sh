export LD_LIBRARY_PATH=.:../../bin/unix64/release:$(LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=.:../../bin/unix64/debug:$(LD_LIBRARY_PATH)
./bmapi_test 0 rx 1000 &
sleep 1
./bmapi_test 1 tx 1000 1