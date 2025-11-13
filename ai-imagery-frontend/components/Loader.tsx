export default function Loader() {
  return (
    <div className="flex flex-col justify-center items-center py-10">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-gray-900"></div>
      <p className="mt-4 text-gray-600">Processing... This may take a minute.</p>
    </div>
  );
}
