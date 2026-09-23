import L from "leaflet";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
import type { Station } from "../lib/types";

// Leaflet's default marker icon paths don't resolve through Vite's bundler without this.
delete (L.Icon.Default.prototype as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow });

const BOGOTA_CENTER: [number, number] = [4.65, -74.1];

export function StationMap({ stations, selected, onSelect }: { stations: Station[]; selected: string | null; onSelect: (stationId: string) => void }) {
  return (
    <div className="h-80 overflow-hidden rounded-lg border border-slate-800">
      <MapContainer center={BOGOTA_CENTER} zoom={11} className="h-full w-full" scrollWheelZoom={false}>
        <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {stations.map((station) => (
          <Marker
            key={station.station_id}
            position={[station.latitude, station.longitude]}
            eventHandlers={{ click: () => onSelect(station.station_id) }}
            opacity={selected === null || selected === station.station_id ? 1 : 0.5}
          >
            <Popup>
              <strong>{station.station_name}</strong>
              <br />
              {station.corridor}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
