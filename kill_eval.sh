for pid in $(ps aux | grep 'eval' | awk '{print $2}'); do
  if ps -p $pid > /dev/null; then
    kill $pid
  else
    echo "Process $pid does not exist"
  fi
done