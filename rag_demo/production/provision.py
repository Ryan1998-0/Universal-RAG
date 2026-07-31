from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from sqlalchemy import select

from rag_demo.production.config import get_production_settings
from rag_demo.production.database import (
    KnowledgeBaseRecord,
    Membership,
    Tenant,
    UserIdentity,
    create_database_engine,
    create_session_factory,
)


@dataclass(frozen=True)
class ProvisionRequest:
    tenant_id: str
    tenant_name: str
    subject: str
    display_name: str
    role: str = "owner"
    knowledge_base_name: str = "主要知識庫"
    profile: str = "default"
    visibility: str = "private"


def provision_identity(session_factory, request: ProvisionRequest) -> dict:
    values = _validated(request)
    with session_factory.begin() as session:
        tenant = session.get(Tenant, values.tenant_id)
        tenant_created = tenant is None
        if tenant is None:
            tenant = Tenant(id=values.tenant_id, name=values.tenant_name, active=True)
            session.add(tenant)
            session.flush()
        else:
            tenant.name = values.tenant_name
            tenant.active = True

        user = session.scalar(select(UserIdentity).where(
            UserIdentity.tenant_id == tenant.id,
            UserIdentity.subject == values.subject,
        ))
        user_created = user is None
        if user is None:
            user = UserIdentity(
                tenant_id=tenant.id,
                subject=values.subject,
                display_name=values.display_name,
                active=True,
            )
            session.add(user)
            session.flush()
        else:
            user.display_name = values.display_name
            user.active = True

        membership = session.scalar(select(Membership).where(
            Membership.tenant_id == tenant.id,
            Membership.user_id == user.id,
        ))
        membership_created = membership is None
        if membership is None:
            membership = Membership(
                tenant_id=tenant.id,
                user_id=user.id,
                role=values.role,
                active=True,
            )
            session.add(membership)
        else:
            membership.role = values.role
            membership.active = True

        knowledge_base = None
        knowledge_base_created = False
        if values.knowledge_base_name:
            knowledge_base = session.scalar(select(KnowledgeBaseRecord).where(
                KnowledgeBaseRecord.tenant_id == tenant.id,
                KnowledgeBaseRecord.name == values.knowledge_base_name,
            ))
            if knowledge_base is None:
                knowledge_base = KnowledgeBaseRecord(
                    tenant_id=tenant.id,
                    owner_user_id=user.id,
                    name=values.knowledge_base_name,
                    profile=values.profile,
                    visibility=values.visibility,
                    active=True,
                )
                session.add(knowledge_base)
                session.flush()
                knowledge_base_created = True

        return {
            "tenant": {"id": tenant.id, "created": tenant_created},
            "user": {"id": user.id, "subject": user.subject, "created": user_created},
            "membership": {"role": membership.role, "created": membership_created},
            "knowledge_base": (
                {
                    "id": knowledge_base.id,
                    "name": knowledge_base.name,
                    "created": knowledge_base_created,
                }
                if knowledge_base is not None
                else None
            ),
        }


def _validated(request: ProvisionRequest) -> ProvisionRequest:
    tenant_id = str(request.tenant_id or "").strip()
    tenant_name = " ".join(str(request.tenant_name or "").split())
    subject = str(request.subject or "").strip()
    display_name = " ".join(str(request.display_name or "").split())
    role = str(request.role or "").strip().lower()
    knowledge_base_name = " ".join(str(request.knowledge_base_name or "").split())
    profile = str(request.profile or "").strip()
    visibility = str(request.visibility or "").strip().lower()
    if not tenant_id or len(tenant_id) > 36:
        raise ValueError("tenant_id must contain 1 to 36 characters")
    if not tenant_name or len(tenant_name) > 160:
        raise ValueError("tenant_name must contain 1 to 160 characters")
    if not subject or len(subject) > 255:
        raise ValueError("subject must contain 1 to 255 characters")
    if not display_name or len(display_name) > 160:
        raise ValueError("display_name must contain 1 to 160 characters")
    if role not in {"owner", "admin", "member"}:
        raise ValueError("role must be owner, admin, or member")
    if knowledge_base_name and len(knowledge_base_name) > 160:
        raise ValueError("knowledge_base_name must contain at most 160 characters")
    if not profile or len(profile) > 64:
        raise ValueError("profile must contain 1 to 64 characters")
    if visibility not in {"private", "tenant"}:
        raise ValueError("visibility must be private or tenant")
    return ProvisionRequest(
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        subject=subject,
        display_name=display_name,
        role=role,
        knowledge_base_name=knowledge_base_name,
        profile=profile,
        visibility=visibility,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision an OIDC identity and tenant.")
    parser.add_argument("--tenant-id", default=os.getenv("RAG_PROVISION_TENANT_ID", ""))
    parser.add_argument("--tenant-name", default=os.getenv("RAG_PROVISION_TENANT_NAME", ""))
    parser.add_argument("--subject", default=os.getenv("RAG_PROVISION_SUBJECT", ""))
    parser.add_argument("--display-name", default=os.getenv("RAG_PROVISION_DISPLAY_NAME", ""))
    parser.add_argument("--role", default=os.getenv("RAG_PROVISION_ROLE", "owner"))
    parser.add_argument(
        "--knowledge-base-name",
        default=os.getenv("RAG_PROVISION_KNOWLEDGE_BASE_NAME", "主要知識庫"),
    )
    parser.add_argument("--profile", default=os.getenv("RAG_PROVISION_PROFILE", "default"))
    parser.add_argument(
        "--visibility",
        default=os.getenv("RAG_PROVISION_VISIBILITY", "private"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = get_production_settings()
    engine = create_database_engine(settings.database_url)
    try:
        result = provision_identity(
            create_session_factory(engine),
            ProvisionRequest(
                tenant_id=args.tenant_id,
                tenant_name=args.tenant_name,
                subject=args.subject,
                display_name=args.display_name,
                role=args.role,
                knowledge_base_name=args.knowledge_base_name,
                profile=args.profile,
                visibility=args.visibility,
            ),
        )
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
