if [ -z "$(git status --porcelain)" ]; then
  # Working directory clean
  echo "Nothing changed"
else
  # Uncommitted changes
  echo " Something changed"
  git status
fi
